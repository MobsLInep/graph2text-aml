# PROGRESS — the Phase 11 GPU matrix on the A4000

Single source of truth for the long-running jobs on the A4000 workstation. Everything needed
to check, resume or restart them is in this file. `DECISIONS.md` holds the *why*
(D-104…D-118); this file holds the *state*. Results are written into `RESULTS.md` §4.9.

**How to use this file (any agent, any session):** read it in full, run the status commands
in §2, compare against the last entry in §6, append a new dated entry to §6, update §5 and
§7, and report a 2–3 line summary to the user. Never rewrite old log entries. If something
failed or stalled: diagnose, write the cause in §8, fix it, and relaunch with the §2 command
(the user said on 2026-10-28 "COMPLETE THE TASK"; spending money — API arms — still needs
an explicit go-ahead). Do not leave a watcher, cron job or polling session behind.

All times are **UTC** (the WSL clock). The user's local time is IST = UTC + 5:30.

---

## 1. Goal

Run every row of the Phase 11 matrix this machine can run, score it, aggregate it, and
record it — for the paper. Done: B1, B2, S1 s42, A1 s42 (Gate 8 pilot: indistinguishable).
Queued: S2, B7, the Gate 8 seeds 1337/2024, B8, A2, A4, A3_F3, A3_F4, B6, A5.
Not runnable here without the user: B3/B4/B5 (API spend), A6 (D-115), Silver, Gold.

---

## 2. Commands

### Check status (read-only)

From Windows PowerShell:

```
(Get-ScheduledTask -TaskName G2T-KeepWSLAlive).State     # must print: Running
wsl -d Ubuntu-24.04 -- tail -n 25 /home/user/g2t-logs/gpu-queue.out
wsl -d Ubuntu-24.04 -- bash /home/user/g2t-run/status.sh
```

`gpu-queue.out` is the queue's own log: one TRAIN/END line per arm, one DECODE pass line,
`score-matrix`/`aggregate` exit codes, `GPU QUEUE DONE` at the end. A `FAILED` line means
that arm failed and the queue moved on (§7 "If something failed").

Per-arm training logs: `/home/user/g2t-logs/train-<ARM>-seed<SEED>.log` (`overfit step`
every ~20 min, `step N` every ~3 h). Decode log: `/home/user/g2t-logs/matrix-decode.log`
(`generated n / 3192` per batch). Matrix state: `artifacts/matrix/matrix_summary.json`.

`distro booted:` in `status.sh` should not change between checks; a fresh boot time means
WSL restarted and the queue died — relaunch it.

### Start / restart (safe to run at any time)

```
Start-ScheduledTask -TaskName G2T-KeepWSLAlive          # no-op if already Running
wsl -d Ubuntu-24.04 -- bash /home/user/g2t-run/launch_gpu.sh
```

It refuses if GPU work is already running or the keep-alive is down. The queue
(`/home/user/g2t-run/gpu_queue.sh`) skips trained arms (`TRAINED` marker), continues a
part-trained arm from its newest `<ARM>_resume.pt` (≤10 steps lost), retries `FAILED`
arms, and skips matrix runs that have a `COMPLETED.json` for the same config hash.

### Rules

Never pass inline `bash -c "…"` with pipes or quotes through `wsl` from PowerShell (zsh
mangles it). Write a script file and run it with
`wsl -d Ubuntu-24.04 -- bash /home/user/g2t-run/g2t.sh bash <file>`. Do not edit files in
`/home/user/g2t-run/` while the queue runs. Do not edit generator source or configs while
an arm is about to start — each arm loads the working tree when it launches.

---

## 3. Configuration

Regime for every trained arm (`/home/user/g2t-run/matrix_regime.txt`, also the matrix
`--generator-override` list, so it is in every run's config hash):

```
generator.attn_implementation=sdpa
training.curriculum=[{name: bronze_only, tiers: [bronze], epochs: 1}]
training.epochs=1
training.per_device_batch_size=1
training.gradient_accumulation=32
training.eval_every=50
training.save_every=50
training.resume_every=10
```

Training adds `generator.generate.decode_test=false` (train now, decode later under one
policy, D-114). Seeds: 42/1337/2024 on S1, A1, B7, S2; 42 elsewhere (D-081). Every seed
reads the seed-42 encoder (D-116). Decoding: greedy, unguarded, batch 8, through the matrix
runner (`11_run_matrix.py --allow-gpu --require-trained --trained-root artifacts/runs/train`);
A5 alone is guarded (four candidates, verification, one repair; D-116). B6 runs
`scripts/09c_local_zero_shot.py` (D-117). After each decode pass: `11c_score_matrix.py`
then `11a_aggregate.py`.

Queue order (`gpu_queue.sh`): train S2 s42 → decode pass (B7 s42, S2 s42, B6) → score →
train S1/A1 ×1337, S1/A1 ×2024, B7 ×1337/2024, S2 ×1337/2024, B8, A2, A4, A3_F3, A3_F4 →
decode pass (everything else, A5 last) → score → aggregate.

Cost per arm-seed on this card: overfit check ~1.7–4 h, training ~20 h, greedy decode
~13 h. Whole queue from 2026-10-28: roughly 3 weeks.

### Code state

Nothing is committed (the user's call). The working tree holds D-104…D-118: generator
fixes, resume, guarded decoding, matrix scoring, executor plumbing, B6, registry changes,
tests (all passing at launch: queue gate 175 passed).

---

## 4. Outputs

| What | Where |
|---|---|
| Trained checkpoints | `artifacts/runs/train/<ARM>_seed<SEED>/<stamp>/<ARM>_final.pt` (+ `TRAINED`) |
| Matrix runs (generations, metrics, per-case) | `artifacts/matrix/<SYSTEM>/seed<SEED>/<hash>/` |
| Tables, figures, missing-run report | `artifacts/metrics/phase11/`, `artifacts/figures/phase11/` |
| Gate 8 pilot | `artifacts/metrics/generator/gate8_pilot_seed42_20261014-1950.json` |
| Gate 8 at three seeds | `make gate8 S1=… A1=…` on the three seeds' `history_*.jsonl` — **not yet run** |

---

## 5. Status

**STOPPED by the user on 2026-11-17. Nothing is running.** The GPU queue ended on
2026-11-09 23:48 UTC. Results so far are in RESULTS.md §4.10.

| Row | State |
|---|---|
| B1, B2, B6 | DONE, scored |
| S1 | DONE at 42, 1337, 2024 |
| A1 | DONE at 42, 2024; **1337 failed** its overfit check |
| B7 | DONE at 42, 1337; **2024 failed** its overfit check |
| S2 | DONE at 42, 1337, 2024 |
| A2 | DONE |
| B8, A4, A3_F3, A3_F4 | **failed** their overfit checks — fix written, not run (D-119) |
| A5 | **failed**: out of memory in the guarded decode — fix written, not run (D-119) |
| A6 | not queued — needs the user (D-115) |
| B3, B4, B5, Silver, κ | not queued — need API spend authorisation |

---

## 6. Log (append-only, newest last)

### 2026-10-05 07:00–11:12 UTC — setup, reproduction, fixes

- Environment built in WSL2 (native Windows cannot run the pipeline — §8 issue 1). Raw data
  fetched and matched to pinned checksums.
- CPU pipeline reproduced exactly: data, cases/splits, facts, Bronze, Bronze evaluation.
- The generator entrypoint had never been executed. Fixed: five crashes in
  `scripts/09_train_generator.py`, encoder/fusion left on CPU, overfit check needing ~24 GB
  and not restoring weights, decoding reading the next token from a pad position and having
  no KV cache, and no test-set decode at all. Recorded as D-104…D-108.
- Preflight on the real 8B (8 cases: train → probes → checkpoint → reload → decode → score)
  passed end to end. Not a result.
- 08:35 `gpu_chain.sh` launched; `encoder-features` finished rc=0 in 509 s.
- 11:12 `pilot_queue.sh` launched; lint + generator gate OK (126 passed); S1 started.

### 2026-10-05 15:13 UTC — S1 overfit check passed

Loss 2.3367 → 0.2499 (step 20) → 0.0121 (40) → 0.0036 (60) → 0.0037 (80) → 0.0002 final;
threshold 0.15. Took 3 h 59 min. Training started 15:13.

### 2026-10-05 18:30 UTC — S1 step 50

Loss 0.0517, fusion gate mean 0.620 (not collapsed), grad(fusion) 5.70e-03.
`S1_step_50.pt` (374 MB) and `samples_S1.jsonl` written. 197 min for 50 steps = 3.9 min per
step with the encoder sweep sharing the GPU.

**Early warning, not a result:** in the two probe cases read (of ten, first 700 characters
each), the narrative generated with the correct graph was identical to the one generated
with a shuffled graph. The model appears to be writing from the text facts in the prompt.
If this holds through training, Gate 8 comes out "indistinguishable" — a negative result to
keep and report (invariant 7). Not yet checked: all ten cases in full, and the attention
mass on the soft tokens (lives in `history_S1.jsonl`, which does not exist yet).

### 2026-10-05 20:58 UTC — PROGRESS.md created

Both jobs alive. S1 between steps 50 and 100. Encoder sweep on run 25 of 27, ~40 min per
run. GPU 90%, 14.4 of 16.4 GB. Run scripts copied to `/home/user/g2t-run/`.

**Pending:** rest of S1 training (~18 h), S1 test decode (est. 3–5 h, never measured at
scale), all of A1 (~27 h), Gate 8 comparison, evaluation; 2 encoder runs, then
`score-cases`, B1/B2, `aggregate`.

Projected: encoder chain done ≈ 2026-10-05 23:30; S1 trained ≈ 2026-10-06 12:45; S1 decoded
≈ 16:00–18:00; A1 and the Gate 8 pilot verdict ≈ 2026-10-07 evening.

### 2026-10-05 21:05 UTC — S1 stopped and relaunched detached (user's decision)

The user wants the runs to survive closing Claude Code, and chose to restart rather than
keep the window open for two days.

- 21:04 attached `pilot_queue.sh` and S1 killed at step 50+ of 328 (≈ 10 h of S1 work
  discarded, including its overfit check). The aborted run directory
  `artifacts/runs/2026-10-05/11-12-43_generator_s1/` and its log
  (`generator-s1-seed42.aborted-1005-2104.log`) are kept; **nothing in them is a result**.
- 21:05 pilot queue relaunched detached (PID 170345): lint + gate OK (126 passed), S1
  started (PID 199442). Confirmed alive after the launching `wsl.exe` exited.
- The encoder sweep was **not** restarted: it is on run 25 of 27 and
  `07_train_encoder.py` cannot skip finished runs (a restart is ~12 h). It stays attached.
  `encoder_tail.sh` (PID 170346, detached) takes over its remaining targets.
- **Not yet proven:** that the WSL distro stays up with no `wsl.exe` attached. It cannot be
  tested while the attached encoder chain holds a session open. Evidence so far: the distro
  rebooted at 08:35 UTC when only a systemd service was running, so systemd services do
  not hold it; `setsid nohup` children of a `wsl` session are expected to. Test when the
  attached chain exits (§7 step 0).

Projected (S1 alone on the GPU once the sweep ends, so possibly faster): overfit check done
≈ 2026-10-06 01:00; S1 trained ≈ 22:00; S1 decoded ≈ 2026-10-07 01:00–03:00; A1 and the
Gate 8 pilot verdict ≈ 2026-10-08 morning.

### 2026-10-06 00:41 UTC — encoder chain ended: sweep done, B2 failed; keep-alive proven

**Keep-alive: proven.** The attached chain exited at 00:37:55. With zero `wsl.exe`
processes on Windows for 100+ s, the distro's boot time was unchanged (`Mon Oct  5
08:35:07 2026`) and S1 (PID 199442) was still running. Claude Code can be closed.

**S1 (restart):** overfit loss 2.3367 → 0.1410 (step 20) → 0.0111 (40) → 0.0054 (60) →
0.0036 (80). On track to pass ≈ 01:00.

**Encoder sweep: finished 23:56, rc 0, 15 h 13 min.** Test AUC-PR, three-seed means,
against the published figures (CLAUDE.md §9):

| | This machine | Published | Reading |
|---|---|---|---|
| GATv2 | 0.857 (0.8511 / 0.8658 / 0.8552) | 0.872 | −0.015, inside ±0.02 |
| MLP control | 0.797 (0.7906 / 0.8092 / 0.7907) | 0.80 | matches |
| **Gate: GATv2 − MLP** | **+0.0605, excludes zero at all 3 seeds → PASS** | +0.070, PASS | **same outcome** |
| GIN | 0.861; GATv2 − GIN −0.003, not significant | 0.880, not significant | same conclusion |
| Graph transformer | 0.896; GATv2 − GT **−0.038, excludes zero at every seed** | 0.888, "not significant" | **CHANGED — see below** |
| SAGE / GCN | 0.760 / 0.712; GATv2 ahead, significant | — | — |
| Ablation: edge features | +0.092, significant | +0.114 | same sign and conclusion |
| Ablation: positional encodings | −0.005, not significant | +0.002, not significant | sign flipped, still null |
| Ablation: focal vs weighted BCE | −0.017, not significant | −0.006, not significant | same sign, still null |

Attention on the laundering path: 0.821 against a 0.478 baseline.

**Finding to carry forward, not variance (D-100 gives conclusions no tolerance):** on this
machine the graph transformer beats GATv2 by 0.038 AUC-PR with the interval excluding zero
at every seed. The published text says it out-scores GATv2 "not significantly", and D-064
keeps GATv2 primary on that basis. D-064 says what evidence would change it — this needs
reading against that entry and a `DECISIONS.md` / `PHASE_LOG.md` entry. **Not yet
written.** S1 and A1 use the GATv2 encoder either way, so the Gate 8 pilot is unaffected.
The positional-encoding sign flip is between two non-significant numbers; reported, and
the "worth nothing" conclusion stands.

**`score-cases`: rc 0** in 40 min. `model_signal` (`gatv2-seed42-epoch44`) written into
16,156 fact records; 10,000 realistic-stream cases have no fact record, as designed.
Bronze was not regenerated.

**B1: completed** — `artifacts/matrix/B1/seed42/ed56a96b84ff/generations.jsonl`, 3,192
cases. Not yet scored (that is `aggregate`).

**B2: FAILED after 13 s.** `ExecutorError: B2 needs the Phase 7 scores at
artifacts/metrics/encoder/scores_gatv2.jsonl; run make score-cases first`. Diagnosis in §8
issue 13: `score-cases` did run and succeeded; **no script in the repository writes that
file.** `FAILED.json` is in `artifacts/matrix/B2/seed42/abde8096675c/`. `aggregate` did not
run. Nothing was restarted.

### 2026-10-11 07:50 UTC — the run is dead; the 10-06 "keep-alive proven" entry was wrong

**State found.** Empty process list. `distro booted:` was the moment of the status command
itself, i.e. the distro had been down. GPU idle. `pilot-queue.out` has no END line and the
S1 log has no traceback: the queue did not fail, it was killed.

**S1's last output:** `00:57:10 overfit check PASSED: 2.3367 -> 0.0032`, `00:57:22 START
training S1`. Its run directory `artifacts/runs/2026-10-05/21-05-12_generator_s1/` holds
only `overfit_check.json` and `run_context.json`. Kept; not a result.

**When and how it died (from the distro's journal and the Windows Security log):**

| Time (UTC / IST) | Event |
|---|---|
| 10-06 03:42:55 / 09:12:55 | Windows service logon (the OpenSSH server spawning a session worker) |
| 10-06 03:43:51 / 09:13:51 | Windows logoff of `sshd_824` — **an SSH session into this Windows machine ended** |
| 10-06 03:44:06 / 09:14:06 | distro: `systemd-logind: The system will power off now!` — a clean shutdown requested by WSL, 15 s later |

No Windows reboot (last boot 2026-09-24), no sleep or resume events, no OOM, no GPU error.
The WSL VM itself stayed up (uptime 16 days); only the distro instance was terminated.

**Cause.** Claude Code was being run over SSH into Windows. A `setsid nohup` process inside
WSL is kept alive by a Windows-side helper (`wslhost.exe`) that belongs to the Windows
logon session which launched it. Ending the SSH session kills that helper; WSL then sees
no client and shuts the distro down after its 15-second idle timeout, taking every Linux
process with it. Detaching inside Linux does not help.

**Why the 10-06 "proof" was wrong.** It checked for `wsl.exe` and waited 100 s. The helper
was still alive because the SSH session was still open, so the test could not fail. The
only valid test ends the Windows session, or kills the helper. "Claude Code can be closed"
was false and the user acted on it.

**Also found:** a scheduled task `KeepWSLAlive` already exists (`wsl.exe -d Ubuntu-24.04
-u root -e /bin/sleep infinity`, boot trigger, runs whether logged on or not — i.e.
outside any SSH session). It is the right mechanism and it is not running: its execution
time limit is 72 h, it last started at boot on 2026-09-24 and was stopped by that limit
(last result 267014). That is why the distro has been shutting down between sessions.

Nothing was restarted. Each status command now boots the distro and so starts
`vllm-8b.service` for a few seconds; harmless while nothing else uses the GPU.

### 2026-10-11 08:40 UTC — user approved both fixes and resume support; pilot relaunched

**Keep-alive.** The user's `KeepWSLAlive` task could not be edited (Windows requires its
stored password). Created a separate task instead: **`G2T-KeepWSLAlive`** — `wsl.exe -d
Ubuntu-24.04 -e /bin/sleep infinity`, boot trigger, S4U logon (no password stored), **no
execution time limit**. Running since 07:54. `KeepWSLAlive` was left untouched.

**Tested with a test that can fail.** "Disconnect" = kill every `wsl.exe`/`wslhost.exe`
that is not the keep-alive's, then make no WSL call.

| Condition | Result |
|---|---|
| Keep-alive stopped, dummy detached job, 60 s | **dummy dead, distro rebooted** — reproduces the 10-06 failure |
| Keep-alive running, dummy detached job, 150 s | dummy alive (16 ticks), boot time unchanged |
| Keep-alive running, **the real S1**, 180 s | S1 alive and at 99% GPU, boot time unchanged (`07:56:36`) |

Still to come: the first check after the user has really disconnected and reconnected.

**Resume support (D-110).** `training.resume_from` continues an interrupted run from
`<run>/<ARM>_resume.pt`, rewritten every 10 steps. Stub: bit-identical to the uninterrupted
run. Real Llama-3.1-8B, interrupted at step 4 of 8 and continued in a second process:
distance to the uninterrupted endpoint 0.000241 against 1.406 moved by steps 5–8 (ratio
0.0002). Resume file 535 MB, 2.5 s to write. `pilot_queue.sh` now picks each arm up where
it stopped.

**B2 fix (D-109).** `score-cases` now writes `artifacts/metrics/encoder/scores_gatv2.jsonl`
through a path shared with the B2 executor. `cpu_tail.sh` launched 08:15: `score-cases` →
`matrix-cpu` → `aggregate`.

**Gates before launch:** ruff clean; 192 tests passed across the generator, matrix, config
and repo-contract modules; the queue's own gate passed (131).

**Launched 08:29:56.** S1 from the start: overfit step 0 loss 2.3367 at 08:33 (the same
value as both earlier attempts). The old S1 log is kept as
`generator-s1-seed42.aborted-1006-0344.log`.

Projected: S1 overfit check done ≈ 12:30; S1 trained ≈ 2026-10-12 10:00; S1 decoded
≈ 13:00–15:00 (decode rate still unmeasured); A1 done ≈ 2026-10-13 20:00; Gate 8 verdict
and evaluation right after.

**Not done:** the `DECISIONS.md` / `PHASE_LOG.md` entry on the graph-transformer finding
(10-06 entry); `PHASE_LOG.md`, and `CLAUDE.md` §9/§10's stale hardware paragraph.

### 2026-10-11 08:45 UTC — check after the user restarted Claude Code

Everything survived. `distro booted:` unchanged (`Sun Oct 11 07:56:36 2026`), keep-alive
task `Running`, same PIDs (queue 69773, S1 99796, `score-cases` 68828), GPU 99%.
S1 is 12 min into its overfit check (next line due ≈ 09:20). `score-cases` at 10,000 of
16,156 records. Whether the restart also ended the SSH session is not known, so this
confirms a Claude Code restart, not necessarily a full disconnect.

### 2026-10-13 06:20 UTC — S1 finished and scored; A1 refused by its overfit check

**Keep-alive confirmed in practice.** `distro booted:` is still `Sun Oct 11 07:56:36 2026`
after two days and at least one Claude Code restart; the task is `Running`. The queue ran
for 41 h unattended and stopped by itself, with an exit code, not by being killed.

**CPU chain (10-11 08:55): all rc 0.** `scores_gatv2.jsonl` has 16,156 rows; B2 completed
in 13.9 s; `aggregate` ran. But `aggregate` reported `NO SYSTEM produced metrics` and wrote
empty tables — §8 issue 18. `RESULTS.md` was not changed by it.

**S1 seed 42 — complete, no interruption, no resume used.**

- Overfit check passed 12:12 (2.3367 → 0.0001). Training 12:12 → 10-12 08:09, 327 steps,
  3.66 min/step, peak 13.75 GB reserved.
- Decode 08:09 → 22:08: **14 h for 3,192 cases** (15.8 s per case at batch 8), against an
  estimate of 3–5 h. Narratives are 135–268 tokens (median 228); none hit the 512 cap.

In-run diagnostics, every 50 steps (ten fixed validation cases; `history_S1.jsonl`):

| Step | Loss | Gate | Probe supported | Shuffled-graph supported | Attention on soft tokens / uniform |
|---|---|---|---|---|---|
| 50 | 0.0483 | 0.621 | 0.9734 | 0.9842 | 0.0006 / 0.0190 |
| 100 | 0.0381 | 0.620 | 0.9891 | 0.9891 | 0.0006 / 0.0177 |
| 150 | 0.0550 | 0.619 | 1.0000 | 1.0000 | 0.0009 / 0.0183 |
| 200 | 0.0339 | 0.619 | 1.0000 | 1.0000 | 0.0008 / 0.0180 |
| 250 | 0.0803 | 0.619 | 0.9948 | 1.0000 | 0.0010 / 0.0174 |
| 300 | 0.0432 | 0.619 | 0.9949 | 1.0000 | 0.0009 / 0.0183 |

**What these say, and it is a negative finding to keep (invariant 7):** the trained S1
does as well with a foreign graph as with its own (gap ≤ 0.011 either way, never in S1's
favour), it pays its sixteen soft tokens about **one-twentieth of the attention a uniform
spread would give them** (lift 0.03–0.055), and the gate never moved from its initial
0.62. The trainer raised `THE SHUFFLED CONTROL TRACKED THIS MODEL` from step 150 on. This
is the within-run control, which is *not* Gate 8 (see `callbacks.py`) — but it is the
outcome the module docstring describes for a model that solves the task from the text.

**Test-set faithfulness, Phase 10 evaluator** (`artifacts/metrics/eval/05-54-44_debug/`,
3,192 test cases, one seed, Bronze-only training, one epoch, unguarded greedy decoding):

| System | Zero-Halluc. | Fact precision | Unverifiable | Coverage | Fact F1 | H2 | H3 | H8 | H9 | Critical |
|---|---|---|---|---|---|---|---|---|---|---|
| Bronze (all 15,707) | 1.000 | 1.000 | 0.000 | 0.860 | 0.924 | 0 | 0 | 0 | 0.918 | 0 |
| B1 (template, test split) | 1.000 | 1.000 | 0.000 | 0.853 | 0.920 | 0 | 0 | 0 | 0.903 | 0 |
| B2 (classifier + template) | 1.000 | 0.903 | 0.097 | 0.853 | 0.877 | 0 | 0 | 1.000 | 0.903 | 0 |
| **S1** | **0.979** [0.974, 0.984] | 0.985 | 0.013 | **0.766** | 0.859 | 0.020 | 0.031 | 0.068 | 0.903 | 0 |

S1 is below the template it was trained to imitate on both headline axes: 2.1% of its
narratives contain a contradicted fact (numeric 2.0%, temporal 3.1%), and it covers 77% of
the salient facts against the template's 85%. H9 is identical to B1's, as D-104 predicted.
Zero critical errors. B2's 0.097 unverifiable and H8 = 1.000 are its appended classifier
sentence, which the checker cannot verify on any case — a property of the baseline, to be
read before it is quoted. Surface metrics are not computed: there is no Gold reference.

**A1 seed 42 — refused by the overfit check after 3 h 37 min.** Step 0 loss **4.9547**
(S1: 2.3367 on the same twenty cases), 0.1264 at step 20, 0.2800, **9.0874** at step 60,
6.6633, final 6.1208. Queue END rc=1; nothing trained.

**Cause, confirmed (D-111, §8 issue 17).** The control wrapper did not forward
`set_target_rms`, so A1's projector was never scaled to the model's embedding RMS
(0.0106): its soft tokens entered the prompt ~100× too large. A1 would have differed from
S1 in two ways, and the comparison would have been biased towards "S1 wins". Evidence:
CPU reproduction (`target_rms` None on the wrapped arm); and on the real model after the
fix, step-0 loss **A1 2.3359, S1 2.3367**, both `target_rms` 0.01057.

**Fix applied and gated, A1 NOT relaunched.** `ShuffledGraphFusion.set_target_rms`
(+ three tests that fail with the fix reverted); ruff clean; 165 tests pass.

### 2026-10-13 06:11 UTC — user approved; A1 relaunched

Launched 06:07:29 with the §2 start command (queue PID 167048, A1 PID 197078). Gate OK
(134 passed); S1 skipped by its `ARM_DONE` marker; A1 from the start. **Overfit step 0
loss 2.3359** — the fixed value, not 4.95. Disconnect simulation repeated against the
running A1 (150 s): alive, boot time unchanged.

Projected: overfit check done ≈ 09:50; A1 trained ≈ 2026-10-14 06:00; decoded ≈ 20:00;
Gate 8 comparison and evaluation minutes after.

**Open question the next check answers first:** whether A1 *passes* the overfit check now.
The scale defect explains the bad start; that the check then converges on a control whose
graph changes every pass is expected (S1 ignores the graph and converges) but not yet
observed.

### 2026-10-15 08:40 UTC — pilot complete: Gate 8 says indistinguishable

**Ran unattended to the end.** `distro booted:` still `Sun Oct 11 07:56:36 2026`; queue
`PILOT QUEUE DONE` at 10-14 19:51:41; no resume was needed for either arm.

**A1 seed 42.** Overfit check passed 09:49 (2.3359 → 0.0001) — the D-111 fix held.
Training 09:50 → 10-14 05:45, 327 steps, peak 13.81 GB; decode 14.1 h. Loss and gate track
S1's almost exactly (step 300: loss 0.0426 vs 0.0432; gate 0.619 both). A1's in-run probes
also show the shuffled graph making no difference and attention on the soft tokens at
0.0006–0.0008 against a uniform 0.018 — the same picture as S1's.

**Gate 8 (`09b_compare_arms.py`, the decision as specified — probe supported rate over the
six aligned diagnostic steps):** S1 0.99487, A1 0.99490, difference −0.00003, tolerance
0.02 → **"S1 and A1 are indistinguishable"**, rc 2. The script's own verdict text reads
"There is no architecture contribution. The paper pivots to the dataset and evaluation
framework." That is the pre-registered consequence; it is **one seed** where CLAUDE.md
specifies three.

**Test set, Phase 10 evaluator** (3,192 cases; one seed; Bronze-only, one epoch; unguarded
greedy):

| System | Zero-Halluc. [95% CI] | Precision | Coverage | Fact F1 | H1 | H2 | H3 | H8 | H9 | Critical |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 (template) | 1.000 | 1.000 | 0.853 | 0.920 | 0 | 0 | 0 | 0 | 0.903 | 0 |
| S1 | 0.979 [0.974, 0.984] | 0.985 | 0.766 | 0.859 | 0.000 | 0.020 | 0.031 | 0.068 | 0.903 | 0 |
| A1 | 0.9715 [0.9655, 0.9771] | 0.985 | 0.778 | 0.867 | 0.002 | 0.023 | 0.033 | 0.074 | 0.903 | 0 |

Paired A1 − S1 on the same 3,192 cases (bootstrap CI, Holm-corrected):

| Metric | A1 − S1 [95% CI] | p (Holm) | Cliff's δ |
|---|---|---|---|
| Zero-Hallucination | −0.0075 [−0.0138, −0.0016] | 0.014 * | −0.008 |
| Fact coverage | +0.0114 [+0.0085, +0.0142] | <0.001 *** | +0.037 |
| Fact F1 | +0.0075 [+0.0054, +0.0096] | <0.001 *** | +0.037 |
| Hallucination rate | +0.0005 [+0.0002, +0.0009] | 0.002 ** | +0.008 |
| Fact precision | −0.0009 [−0.0022, +0.0003] | 0.026 * | −0.004 |
| Unverifiable rate | +0.0004 [−0.0008, +0.0016] | 0.22 ns | +0.005 |

**Reading.** The differences split in direction — S1 hallucinates slightly less, A1 covers
slightly more — every effect size is negligible (|δ| ≤ 0.04), and all are well inside
Gate 8's 0.02 tolerance. Both models pay their graph tokens ~1/20 of uniform attention and
neither changes its output when the graph is swapped. With one seed per arm, a 0.75-point
Zero-Hallucination gap is the size a different training trajectory alone can produce; it
is not evidence the graph is read. **Gate 8 is answered "indistinguishable" for this
regime (Bronze-only, 1 epoch, 16 GB batch-1 settings) at seed 42.** Seeds 43/44 would say
whether the small test-set gaps are stable; they would not change the verdict unless a gap
grew past 0.02.

**Not done:** `PHASE_LOG.md`, `RESULTS.md` and `CLAUDE.md` §9/§10 still describe Gate 8 as
open and the 4 GB machine; the graph-transformer encoder finding (10-06) has no
`DECISIONS.md` entry; `make aggregate` still produces empty tables (issue 18); nothing is
committed.

### 2026-10-15 → 2026-10-28 — matrix plumbing; the first training queue stalled for 12 days

- 2026-10-15: user asked to complete the whole matrix for the paper. Recorded the pilot,
  H9 and the encoder replication (PHASE_LOG, RESULTS §4.9, D-112…D-115). Fixed the
  aggregator gap (`11c_score_matrix.py`, D-112): B1 and B2 now aggregate.
- Decode benchmark on S1's checkpoint: **347 ms per token step at batch 8** (12 s per case);
  batch 32 runs out of memory. The 4-bit base dequantises its weights every step when more
  than one row decodes, so batching buys little. ~13 h per greedy test decode is the cost.
- Found before they ran: the queue's seeds were 43/44 but the registry's are 1337/2024,
  and no encoder exists at those seeds (D-116). Fixed; the queue was replaced mid-run.
- Registry: every trained arm reported raw; A5 = S1 guarded (D-116). Guarded decoding
  written and tested; B6 given its own script (D-117).
- First training queue: B7 seed 42 trained 10-15 15:02 → 10-16 13:54 (327 steps, peak
  13.53 GB). **S2 seed 42 failed its overfit check** (3.23 → 0.43 → 0.023 → **16.38** →
  7.01 → 5.85) and the queue stopped by design. Nobody looked until 10-28: **12 idle days**.

### 2026-10-28 18:30 UTC — S2 fixed, queue rebuilt, relaunched

- Cause of the S2 failure: the overfit check runs AdamW at 1e-3 with no gradient clipping
  while the run clips at 0.3; S2 had memorised the examples (0.023) before diverging. The
  check now clips at the run's own norm (D-118).
- `gpu_queue.sh` replaces the training queue: a failed arm is marked and the queue moves on;
  decode passes run through the matrix with `--require-trained`; scoring and aggregation
  follow each pass.
- S1 and A1 seed 42 imported into the training root and the matrix (D-118). The decode pass
  plans the same run directories, so they will not be re-decoded.
- Launched 18:11:42; gate OK (175 passed); S2 seed 42 training from the start.

### 2026-10-28 19:54 UTC — S2 passes its overfit check with clipping

3.2312 -> 0.0938 (step 20) -> 0.0368 (40) -> 0.0772 (60, **pre-clip grad norm 10.8**) ->
0.0771 (80) -> **0.0341 PASSED**. The same spike that sent the unclipped check to 16.4 at
step 60 shows up here as a gradient norm of 10.8, and clipping contains it - which
confirms D-118's diagnosis. S2 seed 42 is now training (~20 h); the first decode pass
(B7 s42, S2 s42, B6) follows, then the remaining eleven training runs.

Also done this session: S1 and A1 seed 42 scored in the matrix (`11c_score_matrix.py`);
`artifacts/metrics/phase11/` now aggregates four systems (B1, B2, S1, A1) with paired tests
against S1; RESULTS.md, PHASE_LOG.md, CLAUDE.md and DECISIONS (D-112...D-118) updated.

### 2026-10-29 18:04 UTC — S2 seed 42 trained; first decode pass on B7

- Queue alive since 10-28 18:11 (PID 490236 unchanged, keep-alive Running).
- **S2 seed 42 trained**: 327 steps in 10.8 h (shorter prompts, no fact text), peak 11.98 GB.
  Training loss stays at 0.30-0.44 throughout (S1 and B7 sit at 0.03-0.08 with the facts in
  the prompt): with the graph tokens as its only case information the model cannot
  reproduce the narrative. The fusion gate is at 0.620-0.621 again. A first reading of what
  the soft tokens carry; the test-set numbers come from its decode.
- **Decode pass 1** started 05:02. S1 s42 and A1 s42 skipped as already complete (the
  import worked). Seeds 1337/2024 failed fast as not yet trained (expected). **B7 seed 42
  at 3,040 / 3,192** at 18:03 (~2 min per 8 cases): done ~18:40, then S2 s42 (~12 h), then
  B6 (~12 h), then scoring, then the remaining eleven training runs.

### 2026-10-30 05:45 UTC — B7 seed 42 decoded and scored; S2 seed 42 decode nearly done

- Queue alive (PID 490236), keep-alive Running, no reboot.
- **B7 s42** decode completed and scored (by hand, ahead of the queue's scoring step, which
  will now skip it): ZHR 0.9765, coverage 0.7716, fact F1 0.8639, H9 0.903. **B7 − S1:**
  ZHR −0.0025 [−0.0066, +0.0016], coverage +0.0053 [+0.0025, +0.0082]. The text-only arm
  equals the graph arm. Written into RESULTS §4.9.
- **S2 s42** decode at 3,136 / 3,192 (done ~05:55). Then B6 (~12 h), scoring, and the
  eleven remaining training runs.

### 2026-11-17 18:30 UTC — STOPPED by the user; results stored

- The queue had finished on 2026-11-09 23:48 (`GPU QUEUE DONE`); the GPU has been idle
  since. **The user asked to stop and store the results. Nothing was relaunched.**
- Completed and scored: B1, B2, B6; S1 at 42/1337/2024; A1 at 42/2024; B7 at 42/1337; S2 at
  42/1337/2024; A2. Stored in RESULTS.md §4.10 and
  `artifacts/metrics/phase11/results_snapshot_2026-11-17.md`; Gate 8 per seed in
  `artifacts/metrics/generator/gate8_seed{42,2024}_20261117.json`.
- Failed: A1 s1337, B7 s2024, B8, A4, A3_F3, A3_F4 (overfit check: memorised then diverged;
  A4: encoder at the wrong rate) and A5 (OOM in the guarded decode). Fixes written and in
  the working tree (D-119); their stub test run was stopped at the user's request, so they
  are **untested in this state** — run the generator gate before relaunching.
- To resume later: run the generator gate, then the §2 start command. The queue retries the
  six failed arms (~5 days training + ~3 days decoding) and A5.

---

## 7. Next steps

### On the next check

1. Run §2. Read `gpu-queue.out` from the last check's position.
2. **S2 seed 42's overfit check** (first thing the queue does): `overfit check PASSED` in
   `train-S2-seed42.log` after the 2026-10-28 header. If it failed again with clipping, the
   cause is not the optimiser — compare against D-111 (scale) and write it up.
3. For each newly finished training arm: note steps, final loss, gate, attention lift and
   the shuffled-control tracking from the log / `history_<ARM>.jsonl`.
4. After a decode pass: confirm `score-matrix rc=0` and `aggregate rc=0`, read
   `artifacts/metrics/phase11/` and add the new rows to `RESULTS.md` §4.9 (faithfulness,
   H1–H9, paired against S1).
5. When `GPU QUEUE DONE`: run Gate 8 at three seeds (`make gate8` on the S1/A1 histories of
   42, 1337, 2024), write the verdict into RESULTS §4.9 and PHASE_LOG, and report.

### If something failed

`FAILED` in `gpu-queue.out` means one arm failed and the queue went on. Diagnose from that
arm's log, write the cause in §8, fix, and relaunch with §2 — the failed arm is retried
and everything finished is skipped. A decode-pass failure for an arm that is simply not
trained yet is expected in the first pass.

### Waiting on the user

- **API arms** (B3, B4, B5), Silver and the Method A/B κ: need spend authorisation. Run
  `make silver-dry-run` for a cost estimate first.
- **A6**: Qwen3-8B cannot load under the pinned transformers; proposed substitute
  Qwen2.5-7B-Instruct (D-115).
- **Remove `G2T-KeepWSLAlive`** when no more long runs are planned.
- **Commit** the working tree.

---

## 8. Issues seen and fixes applied

| # | Issue | Cause | Fix / state |
|---|---|---|---|
| 1 | Pipeline crashes on native Windows | cp1252 decode; `os.fsync` on a read handle in `utils/io.py::atomic_path` | Run everything in WSL2. **Not fixed in code.** |
| 2 | Stray files in the repo; a `pkill` that killed its own shell | inline `wsl -- bash -c` quoting through zsh | Script files only (§2). Stray files removed. |
| 3 | `HI-Small_Patterns.txt` checksum mismatch | one extra trailing newline in the Kaggle copy | Stripped the byte; matches the pinned digest. |
| 4 | ~7 GB of VRAM taken at every distro boot | the user's `vllm-8b.service` auto-starts | `systemctl --user stop` at the start of each GPU script. |
| 5 | Generator entrypoint crashed five ways in sequence | never executed before; tests drive the harness against a stub | Fixed — D-108. |
| 6 | Overfit check needed ~24 GB and left the weights trained | 20 examples in one pass; no restore | Micro-batched, snapshot/restore — D-106. |
| 7 | Decoding produced garbage on a real model, 120× too slow | right-padding, no KV cache, full-vocabulary logits at every position | Left-pad, KV cache, final-position logits; 8/8 identical to HF `generate` — D-107. |
| 8 | 31 s per pass instead of 7 | batch 2 over-commits 16 GB; WSL spills to shared memory silently | Batch 1 × accumulation 32 — D-105. |
| 9 | Tokenizer test failed on this machine only | depended on the tokenizer *not* being downloadable | Made hermetic. |
| 10 | Pilot queue exited 4 at first launch | its own lint gate caught an unused `noqa` | Removed; relaunched 11:12. |
| 11 | Both jobs slower than their solo estimates | S1 and the encoder sweep share one GPU | Accepted; S1 should speed up when the sweep ends (~23:30). Verify at the next check. |

| 12 | Jobs tied to the Claude Code session that launched them | started as children of `wsl.exe` | Pilot queue relaunched with `setsid nohup` at 21:05 (cost: ~10 h of S1). Sweep left attached; detached tail covers it. |

| 13 | B2 failed: `scores_gatv2.jsonl` missing | `experiments/executors.py:204` reads `<metrics_dir>/encoder/scores_<arm>.jsonl` (keys `case_id`, `score`); `07b_score_cases.py` writes `score_cases.json`, the fact records and `facts.parquet`, and nothing anywhere writes `scores_*.jsonl`. B2 had never been run. | Fixed 2026-10-11 (D-109); re-run in progress — confirm at the next check. |

| 14 | Whole generator queue killed 2026-10-06 03:44 UTC | Jobs were launched from an SSH session into Windows. WSL's Windows-side helper for background processes dies with that session; the distro then idles out in 15 s. The existing `KeepWSLAlive` task would have prevented it but had been stopped by its own 72 h limit since 2026-09-27. Issue 12's fix (detach inside Linux) addressed the wrong layer, and its 100 s test could not fail. | Fixed 2026-10-11: `G2T-KeepWSLAlive` task (no time limit, outside the session), tested with a negative control; plus resumable training (D-110). ≈ 6.5 h of S1 lost, plus 5 idle days. Awaiting confirmation after a real disconnect. |
| 15 | Could not edit the user's `KeepWSLAlive` task | it runs with a stored password, which Windows demands again for any change | Created `G2T-KeepWSLAlive` with S4U logon instead; the original is untouched and still stopped. |
| 17 | **A1 failed its overfit check (4.95 → 6.12)** | `ShuffledGraphFusion` did not forward `set_target_rms`; the generator's `getattr(..., None)` lookup skipped the scaling silently, so the control's soft tokens were ~100× the treatment's. Invisible on the CPU stub, whose embedding RMS is ~1. | Fixed 2026-10-13 (D-111); verified on the real model (step-0 loss 2.3359 vs S1's 2.3367). A1 not yet relaunched. |
| 18 | `make aggregate` reported "NO SYSTEM produced metrics" | The matrix run directories for B1/B2 hold `generations.jsonl` and a completion marker but no metrics; whatever is meant to score matrix generations before aggregation did not run or does not exist. Not diagnosed further. | Fixed 2026-10-15: nothing wrote a run's `metrics.json`; `scripts/11c_score_matrix.py` does, and `make aggregate` runs it first (D-112). |
| 19 | S1's test decode took 14 h, not 3–5 | 15.8 s per case at batch 8; the estimate was never measured at scale | Accepted. A1's decode will take the same. Not investigated. |
| 16 | Resume preflight ran out of GPU memory | the test script built a second 8B model in the process that still held the first | Test split into two processes. Not a defect in the resume code. |

| 20 | Training queue seeds were 43/44; registry says 1337/2024, and no encoder exists at either | queue written from memory, not the registry; entrypoint read the encoder at the generator seed | Fixed before any arm reached them (D-116). |
| 21 | **S2 s42 failed its overfit check; the queue stopped and the GPU idled 12 days** | unclipped AdamW at 1e-3 in the check diverged after memorising (0.023 → 16.4); the queue stopped on any failure | Check clips at the run's norm (D-118); queue moves on past failures. |
| 22 | The orphaned first B7 attempt died after its overfit check when its queue was replaced | not diagnosed; its log ends at the check | B7 retrained from the start by the next queue (~3.6 h lost). |
| 23 | Decode is ~13 h per arm | 4-bit weights dequantised per step at batch > 1 (347 ms/step at batch 8); batch 32 OOM | Accepted; guarded decodes limited to A5 (D-116). |

**Open risks:**

- A Windows reboot, sleep, hibernate or forced update kills every job. The pilot then needs
  the §2 start command; it continues rather than restarting.
- The keep-alive has passed a simulated disconnect, not yet a real one.
- A1's continuation path (`training.resume_from` on the shuffled arm) is covered by the
  shared code and tests but has not been exercised on the real model.
- Probe-evaluation and decode memory after the `empty_cache` changes has only been observed
  at one probe point (step 50, no failure). The full 3,192-case decode at batch 8 is
  unmeasured; if it thrashes or OOMs, lower `generator.generate.batch_size`.
- A5 (guarded decode) has been tested on the stub and its multi-candidate decode against
  HF `generate` on Llama-3.2-1B, but not yet run on the 8B model.
- B6's script has rendered its prompts but not yet run on the 8B model.
