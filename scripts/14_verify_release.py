#!/usr/bin/env python
"""Verify that a stranger can clone this repository and use it.

Runs the release acceptance checks in the order a new user would hit them, and reports
every one -- pass or fail -- rather than aborting on the first. A release that fails one
check and passes eight is a different situation from one that fails all nine, and this
script's job is to say which.

The checks:

1.  **clean-clone**  -- `git archive` the working tree into a pristine directory, so the
    verification never sees an untracked file, a stale build artifact or a populated
    `data/`. This is the check that catches "works on my machine".
2.  **install**      -- `uv sync --frozen` from the lockfile alone.
3.  **quickstart**   -- `scripts/14_quickstart.py`: the real evaluation harness over the
    committed fixture, asserted against `tests/golden/quickstart_evaluation.json`.
4.  **golden**       -- the golden-file tests for the fact layer.
5.  **documented-commands** -- every command this repository documents as runnable without
    data or credentials actually runs. A README whose commands 404 is worse than none.
6.  **script-help**  -- every `scripts/*.py` responds to `--help` with exit 0, except the
    GPU entrypoints, which cannot import torch in the CPU-only environment by design and
    are counted separately rather than failed.
7.  **secret-scan**  -- gitleaks over the **full git history**, not the working tree. A
    secret removed in a later commit is still published by a clone.
8.  **no-data-committed** -- nothing under `data/` or `artifacts/` is tracked, and no file
    in the tree exceeds the large-file threshold.
9.  **licence-separation** -- the CDLA-Sharing-1.0 redistributions carry their NOTICE.

Run it in a container for the real thing (`--in-docker`), or directly for a fast local
check that skips containerisation.

Usage:
    uv run python scripts/14_verify_release.py
    uv run python scripts/14_verify_release.py --in-docker
    uv run python scripts/14_verify_release.py --skip install --json report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Commands the documentation promises work with no dataset, no GPU and no credentials.
#: Each is (label, argv). If you document a new one, add it here -- that is the point.
#:
#: `make test` earns its place: four generator test modules used to import torch at module
#: scope without `pytest.importorskip`, so in the CPU-only environment they failed at
#: *collection* -- which aborts the whole run rather than skipping four modules. `make
#: smoke` is documented as the CI gate and as the second command a stranger types, and it
#: did not work in the environment `make install` produces. Nothing caught it because the
#: development host has the GPU extras installed and CI had never run.
DOCUMENTED_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("make help", ("make", "help")),
    ("make matrix-plan", ("make", "matrix-plan")),
    ("make test", ("make", "test")),
    (
        "scripts/11_run_matrix.py --dry-run",
        (sys.executable, "scripts/11_run_matrix.py", "--dry-run"),
    ),
)

#: Nothing in a clean clone should be larger than this. The corpus, the checkpoints and
#: the raw data all live outside git; a file over the line means one of them leaked in.
MAX_TRACKED_BYTES = 1_000_000

#: Stripped from the environment of every command run inside the clean clone. These point
#: at the *caller's* virtualenv, and a clean-clone verification that resolves against the
#: caller's environment is verifying the wrong thing. uv is explicit about it: it warns
#: "does not match the project environment path `.venv` and will be ignored" and then
#: fails. Caught by the first real CI run, never locally, because a developer shell and a
#: CI job export different subsets of these.
INHERITED_ENV_TO_DROP = frozenset(
    {
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "PYTHONPATH",
        "PYTHONHOME",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
    }
)

#: Written into the clone by the `install` check, not by git. Scanning them would report
#: polars' 100 MB shared object as a leaked artifact, which is the check crying wolf at
#: its own side effect.
NOT_FROM_GIT = (".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__")

#: Top-level module names that live behind an optional extra. `make install` is CPU-only
#: by design, so a script importing one of these cannot answer `--help` in the light
#: environment -- that is the documented design, not a broken script.
OPTIONAL_EXTRA_MODULES = frozenset(
    {
        "torch",
        "torch_geometric",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "vllm",
        "anthropic",
        "streamlit",
        "bert_score",
        "sacrebleu",
        "rouge_score",
    }
)

#: Files that are a licence obligation rather than documentation. Deleting one is a
#: breach, so their presence is a release check.
REQUIRED_NOTICES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "tests/fixtures/NOTICE",
        ("CDLA-Sharing-1.0", "bronze_quickstart.jsonl.gz", "HI-Small_Trans.csv"),
    ),
)


@dataclass
class Check:
    """One release check and its outcome.

    Attributes:
        name: Short identifier, used by ``--skip``.
        description: What a reader should understand the check to mean.
        passed: Outcome. ``None`` while pending, ``False`` when skipped-as-unavailable.
        detail: Human-readable outcome, including the failure reason.
        seconds: Wall-clock duration.
        skipped: True when the check could not run at all (missing tool, `--skip`).
    """

    name: str
    description: str
    passed: bool | None = None
    detail: str = ""
    seconds: float = 0.0
    skipped: bool = False
    output: list[str] = field(default_factory=list)


def run(argv: list[str] | tuple[str, ...], cwd: Path, timeout: int = 1800) -> tuple[int, str]:
    """Run a command and capture its combined output.

    The environment is scrubbed of the caller's virtualenv pointers. Without that,
    ``make test`` inside the clean clone inherits ``VIRTUAL_ENV`` from whatever shell or CI
    job launched the verification, uv sees it disagree with the clone's own project
    environment, warns *"does not match the project environment path"* and ignores it --
    and the check then fails for a reason that has nothing to do with the release. It is
    also the exact opposite of what a clean-clone verification is for: the command must run
    against the clone's environment, never the caller's.

    Args:
        argv: The command.
        cwd: Working directory.
        timeout: Seconds before the command is killed.

    Returns:
        ``(returncode, combined output)``. A timeout returns code 124.
    """
    env = {k: v for k, v in os.environ.items() if k not in INHERITED_ENV_TO_DROP}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            list(argv),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    except FileNotFoundError as exc:
        return 127, str(exc)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def make_clean_clone(dest: Path) -> tuple[bool, str]:
    """Export the tracked-and-stageable tree into a pristine directory.

    Uses ``git archive`` over a temporary stash-free index rather than copying, so the
    result contains exactly what a ``git clone`` would deliver: no untracked files, no
    ``data/``, no ``artifacts/``, no ``.venv``.

    Args:
        dest: Directory to unpack into. Created if absent.

    Returns:
        ``(ok, detail)``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    code, out = run(["git", "rev-parse", "--is-inside-work-tree"], REPO_ROOT)
    if code != 0:
        return False, "not a git repository; cannot build a clean clone"

    # Stage everything into a scratch index so uncommitted work is included exactly as
    # it would be after `git add -A && git commit`. The real index is never touched.
    index = dest.parent / "verify.index"
    env_git = {**os.environ, "GIT_INDEX_FILE": str(index)}
    for argv in (["git", "add", "-A"], ["git", "write-tree"]):
        proc = subprocess.run(
            argv, cwd=REPO_ROOT, capture_output=True, text=True, env=env_git, check=False
        )
        if proc.returncode != 0:
            return False, f"{' '.join(argv)} failed: {proc.stderr.strip()}"
        tree = proc.stdout.strip()
    index.unlink(missing_ok=True)

    tar = dest.parent / "clean.tar"
    proc = subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(tar), tree],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, f"git archive failed: {proc.stderr.strip()}"
    code, out = run(["tar", "-xf", str(tar), "-C", str(dest)], REPO_ROOT)
    tar.unlink(missing_ok=True)
    if code != 0:
        return False, f"tar extract failed: {out.strip()}"

    n = sum(1 for p in dest.rglob("*") if p.is_file())
    for forbidden in ("data/raw", "data/interim", "data/processed", "artifacts/checkpoints"):
        stray = [p for p in (dest / forbidden).rglob("*") if p.is_file() and p.name != ".gitkeep"]
        if stray:
            return False, f"clean clone contains {len(stray)} file(s) under {forbidden}"
    return True, f"{n} files, tree {tree[:12]}, no data/ or artifacts/ payload"


def check_install(clone: Path) -> tuple[bool, str]:
    """Install the light environment in the clean clone from the lockfile.

    ``--extra stats`` matches what `make install-stats`, CI and the CPU image do. It is
    scipy/statsmodels/krippendorff and pulls no torch, and `make test` -- a documented
    command this script then runs -- needs it. Installing without it made the
    `documented-commands` check fail for a reason that was about this script rather than
    about the release.

    Args:
        clone: The clean-clone directory.

    Returns:
        ``(ok, detail)``.
    """
    if shutil.which("uv") is None:
        return False, "uv not on PATH"
    argv = ["uv", "sync", "--frozen", "--group", "dev", "--extra", "stats"]
    code, out = run(argv, clone, timeout=1800)
    if code != 0:
        return False, f"{' '.join(argv)} exited {code}: {out.strip()[-400:]}"
    return True, " ".join(argv[1:])


def check_quickstart(clone: Path, python: str) -> tuple[bool, str]:
    """Run the quickstart in the clean clone and require an exact golden match.

    Args:
        clone: The clean-clone directory.
        python: Interpreter to use.

    Returns:
        ``(ok, detail)``.
    """
    code, out = run([python, "scripts/14_quickstart.py"], clone, timeout=900)
    if code != 0:
        return False, f"quickstart exited {code}: {out.strip()[-600:]}"
    if "QUICKSTART OK" not in out:
        return False, f"quickstart exited 0 without confirming: {out.strip()[-300:]}"
    return True, "220 records scored, matches tests/golden/quickstart_evaluation.json exactly"


def check_golden(clone: Path, python: str) -> tuple[bool, str]:
    """Run the fact-layer golden-file tests in the clean clone.

    Args:
        clone: The clean-clone directory.
        python: Interpreter to use.

    Returns:
        ``(ok, detail)``.
    """
    code, out = run([python, "-m", "pytest", "tests/golden", "-q", "--no-cov"], clone, 900)
    if code != 0:
        return False, f"pytest tests/golden exited {code}: {out.strip()[-500:]}"
    return True, out.strip().splitlines()[-1] if out.strip() else "passed"


def check_documented_commands(clone: Path) -> tuple[bool, str]:
    """Run every command the documentation promises works without data or credentials.

    Args:
        clone: The clean-clone directory.

    Returns:
        ``(ok, detail)``.
    """
    failures = []
    for label, argv in DOCUMENTED_COMMANDS:
        code, out = run(argv, clone, timeout=600)
        if code != 0:
            failures.append(f"{label} -> exit {code}: {out.strip()[-200:]}")
    if failures:
        return False, "; ".join(failures)
    return True, f"{len(DOCUMENTED_COMMANDS)} documented commands ran"


def check_script_help(clone: Path, python: str) -> tuple[bool, str]:
    """Require every pipeline script to answer ``--help`` with exit 0, or say why it cannot.

    A script that cannot describe itself on a clean clone -- because it checks for an
    artifact before parsing its arguments, say -- is undiscoverable to a new user. That is
    the bug this check exists to catch, and it did catch one: `07c_report_tables.py` used
    to exit 1 here because it looked for `encoder_report.json` before reading `sys.argv`.

    **A missing optional extra is not that bug.** `make install` is CPU-only by design
    (CLAUDE.md section 4) and torch lives behind the `graph`/`llm` extras, so the four GPU
    entrypoints genuinely cannot import in the light environment. Failing the release for
    that would be demanding the light environment stop being light. Those are counted and
    reported separately, and **any other non-zero exit still fails**.

    Args:
        clone: The clean-clone directory.
        python: Interpreter to use.

    Returns:
        ``(ok, detail)``.
    """
    scripts = sorted(p for p in (clone / "scripts").glob("*.py"))
    failures: list[str] = []
    gated: list[str] = []
    for script in scripts:
        code, out = run([python, str(script.relative_to(clone)), "--help"], clone, 300)
        if code == 0:
            continue
        # Distinguish "this needs an extra we deliberately did not install" from a script
        # that is simply broken. Only the module name is trusted, not the message text.
        missing = re.search(r"No module named '([\w.]+)'", out)
        if missing and missing.group(1).split(".")[0] in OPTIONAL_EXTRA_MODULES:
            gated.append(f"{script.name} (needs {missing.group(1)})")
            continue
        failures.append(f"{script.name} -> exit {code}: {out.strip()[-200:]}")

    if failures:
        return False, "; ".join(failures)
    detail = f"{len(scripts) - len(gated)} of {len(scripts)} scripts answer --help"
    if gated:
        detail += f"; {len(gated)} need an uninstalled extra ({', '.join(gated)})"
    return True, detail


def check_secret_scan(gitleaks: str | None) -> tuple[bool, str]:
    """Scan the **full git history** for secrets.

    Scanning only the working tree is not enough: a credential committed and later removed
    is still in every clone. This runs against the object database.

    Args:
        gitleaks: Path to the gitleaks binary, or None if unavailable.

    Returns:
        ``(ok, detail)``.
    """
    if gitleaks is None:
        return False, "gitleaks not on PATH — install it or the release is unverified"
    config = REPO_ROOT / ".gitleaks.toml"
    argv = [gitleaks, "detect", "--source", str(REPO_ROOT), "--redact", "--no-banner"]
    if config.is_file():
        argv += ["--config", str(config)]
    code, out = run(argv, REPO_ROOT, timeout=900)
    if code != 0:
        return False, f"gitleaks reported findings over git history: {out.strip()[-400:]}"
    return True, "full git history clean"


def _from_git(clone: Path, path: Path) -> bool:
    """Whether a path in the clone came from git rather than from a later check.

    The `install` check writes a `.venv` into the clone, and that venv contains polars'
    100 MB shared object and pyarrow's test `.parquet` files. Counting those as leaked
    artifacts is the check firing at its own side effect.

    Args:
        clone: The clean-clone root.
        path: The path to classify.

    Returns:
        True when no ancestor of ``path`` is a tool-generated directory.
    """
    return not any(part in NOT_FROM_GIT for part in path.relative_to(clone).parts)


def check_no_data_committed(clone: Path) -> tuple[bool, str]:
    """Assert nothing large, and nothing from data/ or artifacts/, is in the tree.

    Args:
        clone: The clean-clone directory.

    Returns:
        ``(ok, detail)``.
    """
    tracked = [p for p in clone.rglob("*") if p.is_file() and _from_git(clone, p)]
    problems = []
    oversized = [
        (p.relative_to(clone), p.stat().st_size)
        for p in tracked
        if p.stat().st_size > MAX_TRACKED_BYTES
    ]
    for path, size in sorted(oversized, key=lambda x: -x[1]):
        problems.append(f"{path} is {size / 1e6:.1f} MB")
    for pattern in ("*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.parquet", ".env"):
        hits = [p.relative_to(clone) for p in tracked if p.match(pattern)]
        if hits:
            problems.append(f"{len(hits)} file(s) matching {pattern}: {hits[0]}")
    if problems:
        return False, "; ".join(problems)
    total = sum(p.stat().st_size for p in tracked)
    return True, f"no artifact leaked; {len(tracked)} files, {total / 1e6:.1f} MB"


def check_licence_separation(clone: Path) -> tuple[bool, str]:
    """Assert every CDLA-Sharing-1.0 redistribution still carries its NOTICE.

    These files are a licence obligation, not documentation. Deleting one, or adding a
    redistribution without recording it, is a breach.

    Args:
        clone: The clean-clone directory.

    Returns:
        ``(ok, detail)``.
    """
    problems = []
    for rel, required in REQUIRED_NOTICES:
        path = clone / rel
        if not path.is_file():
            problems.append(f"{rel} is missing — this is a licence obligation")
            continue
        text = path.read_text(encoding="utf-8")
        for token in required:
            if token not in text:
                problems.append(f"{rel} does not mention {token!r}")
    if not (clone / "LICENSE").is_file():
        problems.append("LICENSE is missing")
    if problems:
        return False, "; ".join(problems)
    return True, f"{len(REQUIRED_NOTICES)} NOTICE file(s) present and complete"


def verify_in_docker(image: str, tag: str) -> int:
    """Build the CPU image and run this script inside it.

    Args:
        image: Dockerfile path.
        tag: Image tag to build.

    Returns:
        Process exit code.
    """
    print(f"building {tag} from {image} ...", flush=True)
    code, out = run(["docker", "build", "-f", image, "-t", tag, "."], REPO_ROOT, timeout=3600)
    if code != 0:
        print(out[-3000:], file=sys.stderr)
        print(f"FAIL  docker build exited {code}", file=sys.stderr)
        return 1
    print(f"running the verification inside {tag} ...", flush=True)
    code, out = run(
        [
            "docker",
            "run",
            "--rm",
            tag,
            "python",
            "scripts/14_verify_release.py",
            # Inside the image the environment is already installed, and the object
            # database is not present -- the secret scan runs on the host instead.
            "--skip",
            "install,secret-scan",
        ],
        REPO_ROOT,
        timeout=3600,
    )
    print(out)
    return code


def report(checks: list[Check], json_path: Path | None) -> int:
    """Summarise the outcomes and, optionally, write the machine-readable report.

    Args:
        checks: Every check, run or skipped.
        json_path: Where to write the JSON report, or None.

    Returns:
        0 when nothing failed, 1 otherwise. A skip is not a failure -- the summary line
        says how many there were so a green run with six skips cannot read as a clean one.
    """
    print("")
    failed = [c for c in checks if c.passed is False and not c.skipped]
    skipped = [c for c in checks if c.skipped]
    passed = [c for c in checks if c.passed is True]
    print(f"{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "repo": str(REPO_ROOT),
                    "passed": not failed,
                    "n_passed": len(passed),
                    "n_failed": len(failed),
                    "n_skipped": len(skipped),
                    "checks": [asdict(c) for c in checks],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {json_path}")

    if failed:
        print("\nRELEASE NOT VERIFIED:")
        for check in failed:
            print(f"  {check.name}: {check.detail}")
        return 1
    print("\nRELEASE VERIFIED" + (" (with skips)" if skipped else ""))
    return 0


def main() -> int:
    """Run every release check and report the results.

    Returns:
        0 when every non-skipped check passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Verify that a stranger can clone this repository and use it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Checks: clean-clone, install, quickstart, golden, documented-commands, "
        "script-help, secret-scan, no-data-committed, licence-separation.",
    )
    parser.add_argument("--skip", default="", help="comma-separated check names to skip")
    parser.add_argument("--json", type=Path, help="write the machine-readable report here")
    parser.add_argument(
        "--in-docker",
        action="store_true",
        help="build the CPU image and run this verification inside it",
    )
    parser.add_argument("--dockerfile", default="docker/Dockerfile.cpu")
    parser.add_argument("--tag", default="g2t-aml:verify")
    parser.add_argument("--keep", action="store_true", help="keep the clean clone")
    args = parser.parse_args()

    if args.in_docker:
        return verify_in_docker(args.dockerfile, args.tag)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    gitleaks = shutil.which("gitleaks")
    workdir = Path(tempfile.mkdtemp(prefix="g2t-verify-"))
    clone = workdir / "clone"

    checks = [
        Check("clean-clone", "git archive into a pristine directory"),
        Check("install", "uv sync --frozen from the lockfile alone"),
        Check("quickstart", "the documented quickstart, asserted against golden files"),
        Check("golden", "the fact-layer golden-file tests"),
        Check("documented-commands", "every documented no-data command runs"),
        Check("script-help", "every script answers --help with exit 0"),
        Check("secret-scan", "gitleaks over the full git history"),
        Check("no-data-committed", "no data, artifacts or oversized file in the tree"),
        Check("licence-separation", "every CDLA redistribution carries its NOTICE"),
    ]
    by_name = {c.name: c for c in checks}
    python = sys.executable

    print(f"verifying release from {REPO_ROOT}")
    print(f"clean clone -> {clone}\n")

    # Dispatch table rather than a branch ladder: adding a check should be one line here
    # and one Check() above, not a new arm in a function that already does too much.
    # `secret-scan` reads the object database rather than the clone, so it is the one
    # check that still runs when the clean clone could not be built.
    runners = {
        "clean-clone": lambda: make_clean_clone(clone),
        "install": lambda: check_install(clone),
        "quickstart": lambda: check_quickstart(clone, python),
        "golden": lambda: check_golden(clone, python),
        "documented-commands": lambda: check_documented_commands(clone),
        "script-help": lambda: check_script_help(clone, python),
        "secret-scan": lambda: check_secret_scan(gitleaks),
        "no-data-committed": lambda: check_no_data_committed(clone),
        "licence-separation": lambda: check_licence_separation(clone),
    }
    needs_no_clone = {"clean-clone", "secret-scan"}

    try:
        for check in checks:
            if check.name in skip:
                check.skipped, check.detail = True, "skipped by --skip"
                continue
            if check.name not in needs_no_clone and by_name["clean-clone"].passed is not True:
                check.skipped = True
                check.detail = "skipped: clean-clone did not succeed"
                continue

            started = time.monotonic()
            ok, detail = runners[check.name]()
            # The clean clone gets its own interpreter, and everything after `install`
            # must use it -- verifying against the developer's venv would verify nothing.
            if check.name == "install" and ok:
                venv = clone / ".venv" / "bin" / "python"
                if venv.is_file():
                    python = str(venv)

            check.passed, check.detail = ok, detail
            check.seconds = time.monotonic() - started
            mark = "PASS" if ok else "FAIL"
            print(f"  {mark}  {check.name:22s} {check.seconds:6.1f}s  {detail}")

        return report(checks, args.json)
    finally:
        if args.keep:
            print(f"clean clone kept at {clone}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
