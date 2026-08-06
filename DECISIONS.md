# Decision log

Append-only. Newest entries at the bottom. One entry per non-obvious technical choice,
recorded **when the decision is made**, not reconstructed afterwards.

Format:

```
## D-NNN — <title>
**Date:** YYYY-MM-DD · **Phase:** N · **Status:** accepted | superseded by D-MMM
**Decision:** one sentence.
**Rationale:** why this and not the alternative.
**Consequences:** what this forecloses or obliges.
```

---

## D-001 — uv over poetry for dependency management
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** Use `uv` with a committed `uv.lock`.
**Rationale:** Resolution and install are roughly an order of magnitude faster than
poetry, which matters most in CI where we re-resolve on every push; `uv` also manages the
Python 3.11 toolchain itself, so a clean machine needs only `uv` and nothing else.
**Consequences:** Contributors need `uv` installed. The lockfile is authoritative; CI
runs `uv sync --frozen` and will fail rather than silently re-resolve.

## D-002 — Hydra over argparse for configuration
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** All entrypoints are Hydra apps composing from `configs/`.
**Rationale:** The project's core output is an ablation matrix across two substrates,
three corpus tiers, and several fusion variants; Hydra's config groups and multirun
express that grid directly, and its per-run output directory plus saved resolved config
is most of invariant 5 for free. argparse would push the same combinatorics into shell
scripts, where they would go unversioned.
**Consequences:** Config composition is now a testable surface (see
`tests/integration/test_hydra_compose.py`). Interpolations must resolve — a `paths` entry
that does not resolve fails the test suite, not a training run three hours in.

## D-003 — mypy --strict scoped to `facts/` and `eval/` only
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** `mypy` runs in strict mode over `src/g2t_aml/facts/` and
`src/g2t_aml/eval/`, and nowhere else.
**Rationale:** Those two modules are the measurement instrument (invariant 1) — a type
error there corrupts headline numbers silently. Everywhere else, strict typing against
untyped torch/PyG/transformers surfaces buys little and costs a steady stream of
`# type: ignore`. This is a deliberate asymmetry, not an oversight.
**Consequences:** Untyped code elsewhere is tolerated; the two measurement modules have
no escape hatch. If a later phase moves measurement logic out of those directories, the
`files` list in `pyproject.toml` must move with it.

## D-004 — `graph`, `llm` and `human` are optional extras, not core dependencies
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** The default install is CPU-only. CUDA-dependent packages live behind
`--extra graph` / `--extra llm`.
**Rationale:** Phases 1–6 and 10 need no GPU. Forcing a CUDA install into CI would push
it well past the five-minute budget and make it unusable on contributors' laptops.
**Consequences:** CI cannot exercise GPU code paths. Anything in `models/` needs an
explicitly marked (`@pytest.mark.gpu`) test run on a GPU machine before it is trusted.

## D-005 — Torch 2.4.0 / CUDA 12.1 pinned across the graph and llm stacks
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** Pin `torch==2.4.0` in both the `graph` and `llm` extras, with
`torch-geometric==2.6.1` and PyG companion wheels built for that exact torch/CUDA pair.
**Rationale:** PyG companion packages (`torch-scatter`, `torch-sparse`, `torch-cluster`)
are compiled against a specific torch **and** CUDA build; a mismatch surfaces as an
opaque symbol-resolution error at import time. `vllm==0.6.2` independently pins
`torch==2.4.0`, so 2.4.0 is the version that satisfies both stacks without a conflict.
**Consequences:** Upgrading torch means moving PyG, vllm, and the wheel index URL
together, in one commit, with a note here. Do not bump torch alone.

## D-006 — Split manifests are committed files, not seeded runtime code
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** `schemas/splits/<substrate>/{train,val,test}.txt` hold literal ID lists
with a sidecar content hash; `configs/data/*.yaml` reference them and carry no `seed`.
**Rationale:** Invariant 2. A seeded split is reproducible only as long as the upstream
row order, library version, and filtering code all stay fixed — three things that will
not stay fixed across fourteen phases. A committed ID list is reproducible unconditionally.
**Consequences:** Regenerating splits is a visible diff and requires a decision entry.
The split-construction script (Phase 2) is run deliberately, never as part of training.

## D-007 — PyG companion wheels are installed outside the lockfile
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** `torch-scatter`, `torch-sparse` and `torch-cluster` are **not** listed in
the `graph` extra. `make install-pyg` installs them from
`https://data.pyg.org/whl/torch-2.4.0+cu121.html`.
**Rationale:** Their PyPI distributions are sdists that `import torch` in `setup.py`, so
`uv lock` cannot resolve them at all — resolution fails with `ModuleNotFoundError: No
module named 'torch'` regardless of `no-build-isolation-package`, because locking happens
before any environment exists. Building them from source would also produce wheels
compiled against whatever CUDA the build host has, which is the exact drift D-005 exists
to prevent. The prebuilt index wheels are the only correct artifact.
**Consequences:** `uv.lock` does not pin those three; the Makefile does, and the README
records the index URL. When torch moves, the `PYG_WHEELS` variable in the Makefile moves
with it. `uv sync` will not detect a stale or missing PyG install — Phase 7 should assert
the versions at import time.

## D-008 — transformers pinned to 4.45.2, driven by vllm
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** `transformers==4.45.2` rather than the 4.44.2 originally intended.
**Rationale:** `vllm==0.6.2` requires `transformers>=4.45.0`; 4.44.2 made the `llm` extra
unsatisfiable. 4.45.2 satisfies vllm, peft 0.12.0 and trl 0.9.6 simultaneously.
**Consequences:** The `llm` extra's version floor is set by vllm, not by our needs. If
vllm is ever dropped from the stack, this pin can relax.

## D-009 — the local pre-commit hook runs as `language: script`, not `language: system`
**Date:** 2026-08-01 · **Phase:** 0 · **Status:** accepted
**Decision:** `scripts/hooks/check_no_data_staged.py` carries a `python3` shebang, is
executable, and is invoked directly.
**Rationale:** `language: system` with `entry: python ...` fails on any machine where the
interpreter is `python3` and no `python` alias exists — which includes a stock Ubuntu
runner. A shebang'd script has no such dependency.
**Consequences:** The hook script must stay executable and must not import anything
outside the standard library, since it runs outside the project environment.

## D-010 — Polars, not pandas, for the substrate loaders
**Date:** 2026-08-01 · **Phase:** 1 · **Status:** accepted
**Decision:** `src/g2t_aml/data/` uses Polars. pandas stays in the dependency set only for
`utils/io.py`'s Parquet helpers and anything a later phase needs it for.
**Rationale:** HI-Small is 5,078,345 rows and we touch it on every ingest, every split
regeneration and every statistics pass. Polars loads and builds the full account graph in
~3.6 s and computes the complete statistics record in ~9.5 s, against tens of seconds and
several GB for the pandas equivalent — the difference between a loop you can iterate on
and one you avoid running. Two further properties matter more than the speed: Polars keeps
a strict schema, so the zero-padded bank codes cannot be silently coerced to integers; and
its expression API pushes the group-bys the node table needs into a single optimised pass
rather than a Python-level merge.
**Consequences:** `data/` returns `pl.DataFrame`, so every downstream consumer takes
Polars. `utils/io.write_parquet` and `utils/hashing.hash_dataframe` are pandas-typed and
are therefore *not* used by `CanonicalGraph.save`, which writes through
`atomic_path` + `pl.DataFrame.write_parquet` instead — same atomicity discipline, no
conversion. Mixing the two libraries in one module is the thing to avoid; the boundary is
the `data/` package.

## D-011 — AMLworld accounts are keyed `"<bank>|<account>"`, not by account alone
**Date:** 2026-08-01 · **Phase:** 1 · **Status:** accepted
**Decision:** A node identifier in the AMLworld canonical graph is the bank code and the
account identifier joined by `|`.
**Rationale:** Account identifiers are unique only *within* a bank. Keying on the account
alone yields **515,080** nodes; the composite key yields **515,088**, which is exactly the
published figure. Eight account identifiers genuinely collide across banks. Published
statistics settle the question, and they settle it in favour of the composite key —
account-only keying would silently merge eight pairs of unrelated accounts, producing
plausible-looking but wrong degree and component statistics.
**Consequences:** Every downstream identifier carries the bank prefix, so split manifests
(Phase 2) and fact records (Phase 3) inherit it. `|` is the separator because it does not
occur in either field. Any future substrate with the same shape must make the same
decision explicitly rather than defaulting to the bare identifier.

## D-012 — The transaction natural key is built from source text, never from parsed floats
**Date:** 2026-08-01 · **Phase:** 1 · **Status:** accepted
**Decision:** `transaction_key` is assembled from the untyped CSV/patterns text, during
load, before any cast. There is deliberately no function that reconstructs it from a typed
frame; `require_transaction_keys()` raises instead.
**Rationale:** AMLworld rows carry no primary key, so joining the patterns file to the
transactions CSV means rebuilding the natural key on both sides, and that key includes the
amount. Amounts are **not** uniformly two-decimal: 148,151 rows (the Bitcoin ones) carry
six decimals. Any reconstruction that routes the amount through a float and re-formats it
cannot be faithful, because `0.370000` and `0.37` are the same number and different text.
The first implementation formatted to two decimals and lost exactly one transaction —
`unclassified` came out at 1,969 against a published 1,968, with all eight structural
typologies matching. Only the arithmetic identity (3,209 patterned + unclassified = 5,177
flagged) exposed it.
**Consequences:** `load_transactions` always emits `transaction_key`; a frame that loses
the column must be re-loaded, not patched. This is why the golden fixture deliberately
contains spliced six-decimal Bitcoin rows: the regression is cheap to reintroduce and
almost invisible without that case.

## D-013 — One canonical graph representation, carrying an explicit availability mask
**Date:** 2026-08-01 · **Phase:** 1 · **Status:** accepted
**Decision:** Both substrates map into `CanonicalGraph` (node table, edge table, feature
name lists, `AvailabilityMask`, label, typology, provenance), persisted as two Parquet
files plus a JSON sidecar and versioned by `CANONICAL_SCHEMA_VERSION`.
**Rationale:** Everything downstream — fact extraction, corpus generation, the encoder —
should be substrate-agnostic, and the only way that holds is if the substrate differences
are *data* rather than branches. The mask is the mechanism: instead of Phase 3 knowing
"Elliptic2 has no amounts", it consults `availability.monetary_amounts`. `assert_available`
raises `PermissionError` rather than `ValueError` precisely so an invariant-4 breach is
greppable and cannot be swallowed by a generic `except ValueError`.
**Consequences:** Adding a substrate means writing a loader and a mask, not editing
downstream code. Changing the mask's field set is a breaking change to every derived
artifact, so it is versioned alongside the schema. `typology=None` and
`typology="unclassified"` are deliberately different states: None means the substrate has
no typology ground truth, `unclassified` means it has some and reports no match.

## D-014 — The config mask and the code mask are two vocabularies, reconciled by a test
**Date:** 2026-08-01 · **Phase:** 1 · **Status:** accepted
**Decision:** `configs/data/*.yaml` keeps its eight-key `availability` block (asserted by
`REQUIRED_AVAILABILITY_KEYS` since Phase 0); `AvailabilityMask` has nine finer-grained
fields; `to_config_mask()` projects one onto the other and
`test_config_mask_matches_the_code_mask` asserts they agree for both substrates.
**Rationale:** The Phase 0 config vocabulary predates the dataclass and is load-bearing in
existing tests, while the dataclass needs distinctions the config lacks — notably
`absolute_timestamps` versus `fine_temporal_resolution`, which is what actually forbids
"within 22 hours" on Elliptic2. Collapsing to one vocabulary would either lose that
distinction or churn Phase 0's tests. Two vocabularies is a tolerable cost; two
vocabularies that *disagree* is not, since a fact could be licensed by one and forbidden by
the other.
**Consequences:** Any new mask field needs a decision about its config projection.
`account_ids` has no dataclass counterpart and is hardcoded True, since both substrates
carry node identifiers.

## D-015 — Two Phase 0 availability flags were wrong and have been corrected
**Date:** 2026-08-01 · **Phase:** 1 · **Status:** accepted
**Decision:** `amlworld.entity_types` true → **false**; `amlworld.node_features` true →
**false**; `elliptic2.node_features` true → **false**.
**Rationale:** The Phase 0 masks were written before any data was on disk and were
optimistic. Reading the actual CSV header settles the AMLworld cases: the schema carries a
bank code and nothing that says "mixer" or "exchange", and AMLworld ships no node table at
all — our node attributes are aggregates we derive. The Elliptic2 case is a
misreading rather than a data question: its feature columns do exist, but they are
anonymised, and the mask governs what may be *asserted*, not what is present. An anonymised
column licenses no assertion, so recording it as available was wrong.
**Consequences:** No narrative on either substrate may name a business type. The
`semantic_node_features` docstring states explicitly that the flag does not forbid
statements about derived aggregates such as degree or total sent — those are licensed by
`monetary_amounts` and the edge data — so Phase 3 does not over-mask.

## D-016 — Published statistics are asserted at ingest, and disagreement aborts the run
**Date:** 2026-08-01 · **Phase:** 1 · **Status:** accepted
**Decision:** `scripts/01_ingest.py` compares observed node, edge and per-typology counts
against the published figures and raises `IngestError` on any mismatch. Substrates that
cannot satisfy this (the test fixture) set `data.verify_published: false` explicitly, and
subsetted runs record `subsetted_n_rows` in the manifest.
**Rationale:** Dataset statistics that silently disagree with the cited paper are the
classic way a data phase goes wrong, and they are near-invisible once downstream numbers
depend on them. Asserting at ingest converts that into a loud, immediate failure. The
comparison table is still computed and written in the disabled cases, so the mismatch is
visible in the statistics report — what is switched off is the abort, not the reporting.
**Consequences:** A deliberate change to the loader that shifts a count requires updating
`PUBLISHED_STATISTICS`/`PUBLISHED_TYPOLOGY_COUNTS` and this log. CI cannot run these checks
(the data is 475 MB and not in the repo), so `tests/integration/test_published_statistics.py`
skips when the data is absent and the ingest script is the real enforcement point.

## D-017 — Self-loops and multi-edges are kept, not cleaned
**Date:** 2026-08-01 · **Phase:** 1 · **Status:** accepted
**Decision:** Phase 1 performs no sampling, filtering or deduplication. HI-Small's 591,212
self-loops and 561,575 multi-edge node pairs are carried into `data/interim` intact.
**Rationale:** They are real properties of the substrate, and 11.6% of edges is far too
large a share to remove silently — doing so would change every degree, component and class
statistic, and the changed numbers would no longer reconcile with the published ones.
Cleaning is a modelling decision, and modelling decisions belong to the phase that makes
them, with its own entry here.
**Consequences:** Phase 7 must decide explicitly whether the GAT keeps self-loops (PyG adds
its own by default, which would double them) and whether it treats the graph as a
multigraph. `structural_statistics` reports both counts on every ingest so the decision is
made against real numbers.

## D-018 — Case construction: 2-hop, n_max 150, amount-descending prune, laundering paths preserved
**Date:** 2026-08-01 · **Phase:** 2 · **Status:** accepted
**Decision:** An AMLworld case is built by `extract_case(seed, window, k_hops=2, n_max=150,
prune_rule="amount_desc", preserve_laundering_paths=True, seed=1337,
max_neighbours_per_node=64)`. Elliptic2 cases are pass-through — a provided labelled
subgraph *is* the case, recorded as `extraction_method="provided"`.
**Rationale:** Each parameter earns its value.
*k=2* reaches the counterparty-of-a-counterparty, which is the shortest radius at which a
two-sided typology (scatter-gather, gather-scatter) is expressible at all; k=1 cannot
represent one, and k=3 multiplies case size without adding a typology. The sensitivity
table measures both claims rather than asserting them.
*n_max=150* is far above the largest AMLworld stream (32 transactions) plus its context, so
pruning is a guard against degree skew rather than a routine step — it fires on 3% of
cases at k=2.
*amount_desc* is the rule a financial-crime analyst would apply unprompted: when a case
must be cut down, the largest movements are the ones that stay. `recency` and `degree` are
implemented so the sensitivity analysis can show the choice is not carrying the result.
*preserve_laundering_paths* is the one non-negotiable. A pruner that severs the laundering
path returns a case labelled suspicious with no evidence in it, and every narrative
generated from that case would assert a scheme it cannot show. Preservation runs before
the budget opens and may overrun `n_max`; when it does, the case records `n_max_exceeded`
rather than dropping evidence silently.
*max_neighbours_per_node=64* is not in the original protocol and was added because
HI-Small's maximum out-degree is 168,672 against a median of 2 (Phase 1). One uncapped hop
through a hub produces a case an order of magnitude past the node budget and dominated by
an account unrelated to the seed. 64 is four times the widest fan AMLworld generates
("Max 16-degree Fan-Out" in the patterns file), so it cannot truncate a synthetic typology.
Every case records whether the cap bound.
*seed* is recorded and folded into the case identifier but never drawn from: extraction is
fully deterministic, and every ordering decision is a total order over edge *content*, so a
reordered input frame yields a byte-identical case. Randomness lives only in sampling.
**Consequences:** Changing any parameter changes every case identifier, and therefore
invalidates the committed split manifests — which is the intended cost. `case_nodes` and
`case_edges` are stored as positions into the interim graph rather than as copies, so the
corpus is 40x smaller but is only meaningful against the graph it was cut from; the
collection records that graph's hash and refuses to materialise against another.

## D-019 — The case window is capped at 48 hours, and that cap is what makes a temporal split possible
**Date:** 2026-08-01 · **Phase:** 2 · **Status:** accepted
**Decision:** A case window is the seeding activity's temporal extent plus 12 hours on each
side, capped at **48 hours**. A stream too long for the cap gets a 48-hour review window
centred on its median transaction time.
**Rationale:** This was forced by a measurement, not chosen for elegance. HI-Small's
laundering streams have a median span of 74 hours and reach 202, and they all start within
the first twelve days of an 17.7-day substrate. A temporal split needs each split to
occupy a band at least one case-window wide — otherwise no case fits inside it — so three
bands plus buffers must fit inside those twelve days. With uncapped windows (median 98h
padded) the val band is narrower than a single case and **comes out empty**; that is the
first thing the pipeline did. Measured retention and val viability across the cap:

| cap | retained | achieved ratio | stream transactions kept | streams kept whole |
|---:|---:|---|---:|---:|
| none | 44% | 0.84/0.12/0.04 | 100% | 100% |
| 96h | 50% | 0.78/0.10/0.11 | 92% | 59% |
| 72h | 56% | 0.79/0.09/0.12 | 82% | 42% |
| **48h** | **51%** | **0.70/0.17/0.14** | **65%** | **28%** |

48 hours is the only setting that yields a usable three-way split at close to the requested
proportions. The cost is real and is stated rather than hidden: a case built from a long
stream contains 65% of that stream's transactions on average, and only 28% of streams are
covered in full. This is defensible on its own terms — an analyst reviews a *period*, not a
scheme's entire lifetime — but it is a cost, and the sensitivity table reports it.
**Consequences:** Phase 3's fact layer must not assume a case contains its whole stream.
`typology` on a case means "this case is part of a stream of this typology", not "this case
exhibits this typology in full". Any claim in a narrative about the *completeness* of a
scheme is unsupported and the verifier must reject it.

## D-020 — Split boundaries are searched, not taken from a quantile
**Date:** 2026-08-01 · **Phase:** 2 · **Status:** accepted
**Decision:** The two boundaries are chosen by evaluating every pair on the 24-hour snap
grid against the actual case population, scored on `|achieved - target| + 1.5 x (1 - kept)`,
subject to every split holding at least 5% of survivors. Requested proportions are
70/15/15; **achieved** proportions are recorded in the manifest.
**Rationale:** The obvious rule — boundaries at the 70th and 85th percentile of case start
— does not work here, and its failure is silent. Cases have duration, so how many survive a
boundary pair depends on where the whole population's intervals sit relative to *both*
boundaries. On HI-Small those two percentiles fall about a day apart, and a one-day band
cannot hold a two-day case: the quantile rule produces an empty val split. Searching the
grid directly optimises the quantity actually wanted and adapts to a population whose
geometry is not known in advance. The retention term exists because scoring on proportion
error alone buys an exact 70/15/15 by discarding most of the corpus, which is a worse split
than a slightly uneven one built from twice as many cases.
**Consequences:** The split is not exactly 70/15/15 and never will be on this substrate.
Reporting the target as though it were achieved would be the dishonest option, so the
manifest carries both, and the paper must quote the achieved figures.

## D-021 — Node overlap is reported, not enforced; strict mode exists but is not the default
**Date:** 2026-08-01 · **Phase:** 2 · **Status:** accepted
**Decision:** `overlap_mode: report`. The node overlap rate is measured, published, and
recorded in the manifest and the audit report. `strict` mode, which drops every test case
sharing a node with a train case, is implemented and available.
**Rationale:** The measured rate on HI-Small is **65.4%** of test cases. That number looks
alarming and is mostly a property of the substrate: 72.2% of accounts sit in one giant
weakly-connected component (Phase 1), so a two-hop neighbourhood around almost any account
reaches accounts that some training case also reached. The recurring accounts are
overwhelmingly high-degree intermediaries — the correspondent-bank equivalent — which is
exactly what a real institution's data looks like and is not information about the test
label. Strict mode on a graph this dense removes most of the test set and biases what
remains toward isolated, atypical activity, which would make the test set *less*
representative, not more. What genuinely matters is whether the same *event* appears on
both sides, and that is covered by two separate checks that are far stricter: edge overlap
(the same transaction) and stream atomicity (the same laundering scheme), the latter being
a hard failure.
**Consequences:** The paper must publish the 65.4% figure alongside the edge-overlap rate
and state the reasoning, rather than leaving a reviewer to discover it. If a reviewer
insists on node-disjointness, `overlap_mode=strict` regenerates the manifest; the resulting
test-set size must then be reported too.

## D-022 — Six-hour buffer, because the straddle rule already does the work
**Date:** 2026-08-01 · **Phase:** 2 · **Status:** accepted
**Decision:** `buffer_hours: 6.0`. Any case whose window straddles a boundary is dropped
outright; the buffer additionally drops cases whose window starts or ends within six hours
of one.
**Rationale:** The straddle rule is not a heuristic — it is exact. A case that does not
straddle a boundary contains no transaction on the far side of it, so there is no bleed to
eliminate and the buffer is margin, not mechanism. Margin is still worth having against
minute-resolution timestamps and any future change to how a window is derived, but every
hour of it costs cases: raising the buffer from 6h to 24h on this corpus drops a further
~9% for no measurable gain in disjointness. Six hours is a quarter of the snap grid.
**Consequences:** Recorded in the manifest as `split_params.buffer_hours` and reported in
the drop tally as `within_buffer`, so the cost is always visible next to the benefit.

## D-023 — The realistic-imbalance stream reports the prevalence it observes
**Date:** 2026-08-01 · **Phase:** 2 · **Status:** accepted
**Decision:** The second test stream samples seeds **uniformly** from accounts active in
the test window, with no matching, no stratification and no laundering filter. The
prevalence that emerges is measured and recorded. A `target_prevalence` knob down-samples
suspicious cases when one is wanted, and records that it did.
**Rationale:** The observed case-level prevalence is **7.3%**, not the ~0.1% the
transaction-level rate (1 in 981) would suggest. That gap is not an error, it is the most
interesting thing the stream measures: a case is a two-hop neighbourhood over 48 hours, so
it aggregates hundreds of transactions, and the probability that *at least one* of them is
flagged is far higher than the probability that any given one is. Reporting 7.3% is the
honest number for this case definition. Forcing 1-in-500 by discarding 97% of the
positives found would be choosing the answer, and at 10,000 cases it would leave roughly
20 suspicious cases — too few to support a stable metric.
**Consequences:** The "validation in a realistic decision setting" claim must be stated in
terms of *case-level* prevalence and must quote both numbers, since a reader will assume
the transaction-level one. If a reviewer wants a bank-like alert rate, `target_prevalence`
provides it, but the stream then has to be built much larger for the positive count to
support a metric — the arithmetic belongs in the paper, not in a silent default.

## D-024 — Hard negatives are mined per window, and ordinary negatives are drawn at random
**Date:** 2026-08-01 · **Phase:** 2 · **Status:** accepted
**Decision:** The hard-negative share is taken inside each case window rather than across
the pool as a whole, and the ordinary negatives are a uniform random draw from what is
left rather than the next-highest scorers.
**Rationale:** Both were found by looking at the built corpus, and both would have been
invisible in aggregate. Mining the top scorers globally concentrates hard negatives
wherever high-motif licit activity happens to fall in time; the temporal split then
inherits that clumping. Measured on HI-Small: a globally-mined corpus at 28.9% hard
negatives overall came out **24.8% in train and 64.0% in test**. A test split with two and
a half times the train split's share of the hardest population produces test numbers that
cannot be compared to validation numbers, and the aggregate rate — which passes every gate
— says nothing about it. Taking the same share inside each window makes the rate uniform
across time by construction, so wherever the boundary falls it cuts a representative
population. Separately, an "easy" negative chosen for being the next-best scorer is not
easy: taking the top of the remainder compresses the contrast between the two populations,
which is the only thing having both is for.
**Consequences:** The hard-negative rate is now approximately uniform across splits, and
the per-split rate is checked at the gate rather than only the corpus-wide one. Windows
with too few licit candidates to fill their quota contribute proportionally fewer hard
negatives rather than borrowing from elsewhere, so the pool must be oversampled enough for
every window to fill — which is what `hard_negative_oversample` is sized for.

## D-025 — The fact record represents absence with a typed sentinel, never 0 and never a bare null
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** Every fact family a substrate may not support is a union with
`facts.schema.Unavailable`, a distinct class carrying a machine-readable `reason` and
serialising to `{"available": false, "reason": ...}`. `0`, `0.0`, `""` and bare `None` are
never used to mean "this substrate cannot say".
**Rationale:** Invariant 4's failure mode is quiet. A missing amount defaulted to `0.0`
reads downstream as "nothing moved"; a missing timestamp defaulted to `None` reads as
"unknown, probably fine". Both are assertions the substrate does not license, and both
produce a fluent, plausible, wrong sentence three phases later. A distinct *type* cannot be
mistaken for either: a consumer that forgets to branch gets a `TypeError` at the point of
the mistake rather than a wrong narrative at the end of the pipeline. The sentinel is also
falsy and value-equal on its reason, so `if is_available(x)` reads naturally and two
sentinels for the same cause compare equal.
There is one deliberate exception. `labels.min_hops_to_known_illicit`, `motifs.cycle.length`
and the other motif descriptors use a bare `None`, because there `None` is a *measured
value* — "no illicit node is reachable", "there is no cycle" — rather than a masked field.
The checker treats the two differently and must: a claim against a measured null is a claim
about something the data did answer, while a claim against a sentinel is a claim about
something it cannot address. Collapsing them would put a compliance-dangerous unsupported
assertion in the same bucket as an ordinary arithmetic slip.
**Consequences:** Every consumer of a fact record branches on availability before reading a
value, and `mypy --strict` over `facts/` enforces it — a union cannot be dereferenced
without narrowing. The serialiser renders the two absences differently on purpose, so the
B7 baseline is not handicapped by being unable to tell them apart.

## D-026 — Burst detection: >= 5 transactions within <= 24 hours, reporting the tightest such window
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** A burst exists iff some window of at most `burst_window_hours` (24) contains
at least `burst_min_transactions` (5) transactions. The burst *reported* is the tightest
qualifying window: maximise the transaction count, then among windows achieving that
maximum, minimise the span. `temporal.burst_window_hours` records the **observed** span of
that window, not the configured cap. Implemented as a two-pointer sweep over sorted
timestamps, O(n log n), with a total order over `(count, -span, start)` so two runs cannot
disagree about which of several equally tight windows to report.
**Rationale:** "Burst" must not mean "many transactions". Forty transactions spread evenly
over two days is not bursty and six inside twenty minutes is, and a detector that could not
separate those would make `rapid_dispersal` fire on any active account. `N = 5` sits well
inside a real AMLworld pattern — its fans run to 16 counterparties and its streams to 32
transactions — while excluding the median case, which holds 11 transactions across its whole
48-hour window. `H = 24` is set equal to the vocabulary's temporal scale so the qualitative
phrase and the structural detector cannot disagree about what "short" means.
**One consequence was found by running it and is worth stating.** Because
`burst_window_hours` is bounded above by `H` by construction, a `rapid_dispersal` binding of
`"< 24"` would be satisfied by *every burst the detector can possibly report* — measured on
HI-Small, the maximum-count burst routinely occupies 20+ hours, which no reader would call
rapid. A descriptor that always holds is not a claim. The binding is therefore `"<= 6"`,
strictly tighter than the detection window, and
`tests/unit/test_facts_coverage.py::test_burst_descriptor_threshold_is_strictly_tighter_than_the_detector_cap`
plus its config-aware twin enforce the relationship so an override cannot reintroduce the
vacuity.
**Consequences:** Any future descriptor bound to `temporal.burst_window_hours` inherits the
strictness requirement. Changing `burst_window_hours` without re-checking the bindings below
it is a test failure, by design.

## D-027 — Tolerance policy, per claim type, published
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** The checker's tolerances are fixed as follows and go in the paper:

| Claim type | Tolerance |
|---|---|
| Counts | **Exact.** No latitude at all. |
| Monetary amounts | **1% relative**, with a 0.01 absolute floor. |
| Durations | Within **one unit of the granularity the narrative itself states**. |
| Categorical | Exact, against the controlled vocabulary. |
| Qualitative | Resolved through the risk-descriptor binding table, else UNVERIFIABLE. |
| Entity references | Must appear in `entity_inventory.node_ids`, else H1. |
| Regulatory | Must match a whitelisted reference, else H6. |

**Rationale:** Each row is a different judgement and none generalises to the others.
*Counts exact*, because "nine accounts" when there are eight is wrong in a way no rounding
convention redeems, and there is no way to write it that makes it right. `counts_exact` is
not a knob — `ToleranceConfig` refuses to construct with it disabled, because a run that
relaxed it would report a number that does not mean what the paper says it means.
*Monetary 1% relative rather than exact*, because "approximately USD 482,000" against a
record of 482,300.00 is what a good first draft says: an investigator wants a magnitude, and
a checker that marked good writing unfaithful would push the corpus toward stilted precision.
The absolute floor exists because 1% of a 0.01 BTC transfer is 0.0001, which no written
rounding of a number can hit — and HI-Small has 148,151 six-decimal Bitcoin rows (D-012).
*Durations within one unit of the STATED granularity* is the subtle one, and it is
deliberately asymmetric. "About three days" against 76 hours is SUPPORTED, because the
narrative claimed a precision of one day and is right at that precision. "76 hours" against
80 hours is CONTRADICTED, because it claimed a precision of one hour and missed by four. The
identical four-hour error resolves differently depending on how precisely it was stated,
which is the only rule that neither punishes appropriate vagueness nor waves through a
genuine error. The granularity is read from the claim (`DurationClaim.unit`) rather than
imposed by the checker.
**Consequences:** The tolerance table is a published commitment, so a reviewer can re-derive
our faithfulness number. `configs/facts/default.yaml` carries the same values and
`tests/integration/test_facts_config_contract.py` asserts the two agree, so a run cannot be
configured one way and recorded another.

## D-028 — UNVERIFIABLE is a first-class verdict and is never collapsed into the other two
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** `Verdict` has three members and the reporting layer publishes all three rates
separately, alongside a Critical Error Rate over H4/H6/H7 computed independently of overall
faithfulness. A claim the checker cannot resolve returns UNVERIFIABLE — never SUPPORTED.
**Rationale:** A binary split has to place "the narrative says the account received USD
480,000 and it received USD 480,000" and "the narrative says the account is registered in
Panama, about which the substrate is silent" into the same two buckets, and both available
answers are wrong. Calling the second faithful licenses every unsupported assertion a model
can invent; calling it unfaithful equates a hedge with a falsehood.
UNVERIFIABLE is also the diagnostically most valuable bucket, which is why it must not be
merged away for a tidier headline number. It collects exactly the compliance-dangerous
claims: assertions about masked facts, unsupported attributions, vague intensifiers that
resolve to no measurement. A system with high SUPPORTED *and* high UNVERIFIABLE is not a good
system — it is one that has learned to say impressive things the graph cannot back, and that
is the failure mode this project exists to detect. The critical three are separated for the
same reason: averaging a "mixer" attribution into a percentage with a rounded amount lets a
system with a 2% critical-error rate look identical to one with 0%, and the difference between
those is the difference between deployable and not.
**Consequences:** Leniency is a bug. Where a claim is hard to resolve the answer is
UNVERIFIABLE, and the test suite asserts it at the boundaries where the temptation is
strongest — an unregistered field path, a malformed claim value, a bool passed where a
number is expected, a descriptor bound to a masked field.

## D-029 — Entity-type vocabulary is excluded, and the exclusion is the mechanism
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** The controlled role vocabulary is `originator`, `intermediary`, `beneficiary`,
`pass_through`, `terminal`, `hub` — and contains no business-type term. `mixer`, `tumbler`,
`exchange account`, `shell company`, `darknet market`, `casino account`, `money service
business` and `hawala` are on the **forbidden** list as hallucination class H4, severity
Critical.
**Rationale:** Neither substrate carries an entity-type column. AMLworld's schema has a bank
code and nothing that says what an account *is*; Elliptic2's node features exist but are
anonymised. Both masks therefore set `entity_types=false` (D-015). Any business-type claim is
unevidenced by construction, on every case, in every run.
The decision worth recording is not that such claims are wrong but *where they are stopped*.
Calling an address a "mixer" is the single most plausible-looking sentence an unconstrained
model will write about a Bitcoin subgraph — it is fluent, it is the kind of thing a real SAR
contains, and it is exactly what a reader would fail to question. Detecting it after
generation would work; excluding the words from the vocabulary the system may draw on stops
it being written. That is why the exclusion is in `vocab_v1.yaml` and enforced by
`test_entity_type_terms_are_absent_from_the_role_vocabulary`, rather than being left to the
checker alone. The checker still catches it (H4, via `check_narrative_text`), because
defence in depth is appropriate for a Critical class.
**Consequences:** No component may name a business type on either substrate, and adding one
would require a substrate that carries the data plus an entry here. The six roles are
defined by in-case degree bindings published in the vocabulary, so a role claim is checkable
against a number rather than against an intuition.

## D-030 — `cross_border` is in the schema as a permanent sentinel, not omitted
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** `flow.cross_border` exists in `case_facts_v1.json`, is typed as
`$defs.unavailable`, and is emitted as `{"available": false, "reason":
"no_substrate_carries_jurisdiction"}` on every case of every substrate. A separate
`flow.cross_institution` boolean carries what *is* derivable — whether the case spans more
than one bank code.
**Rationale:** This deliberately bends the rule "do not add a field you cannot compute for at
least one substrate", and the reason is that omitting it is worse. Neither AMLworld nor
Elliptic2 carries jurisdiction, so cross-border movement is unevidenced — but "cross-border"
is a phrase a SAR narrative reaches for constantly, and a field that simply did not exist
would mean such a claim matched nothing and was silently dropped rather than counted. Present
as a permanent sentinel, the claim resolves to UNVERIFIABLE and appears in the report. The
field's job is to make a specific unsupportable claim *visible*, not to hold a value.
Splitting out `cross_institution` matters independently: multi-bank movement is real, is
derivable from the node table's bank codes, and is a different claim from crossing a border.
Conflating them would let a genuine finding license a fabricated one.
**Consequences:** `cross_border` is the one field with a checker that returns UNVERIFIABLE
unconditionally (`check_unavailable_only`). If a future substrate carries jurisdiction, this
entry is superseded and the field's type changes — a schema-version bump.

## D-031 — `facts/motifs.py` and `data/motifs.py` are two modules and share no code
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** The Phase 3 detectors live in `src/g2t_aml/facts/motifs.py` and return
`present: bool` plus quantitative descriptors against recorded thresholds. The Phase 2 soft
scorer in `src/g2t_aml/data/motifs.py` is untouched and continues to return continuous [0,1]
similarity. Neither imports the other.
**Rationale:** They answer different questions and a shared implementation would force one
set of thresholds to serve both. The scorer asks "how much does this case *resemble* a
typology", so hard-negative mining can rank licit cases by how deceptive they look; it needs
a smooth ranking and no thresholds at all. The detector asks "does this case *contain* a
fan-out, and how wide is it", because a narrative quotes a width and a checker verifies one.
A soft score cannot go in a sentence.
The real risk is directional: if the two shared code, a threshold changed to improve
hard-negative mining — a Phase 2 concern with no bearing on measurement — would silently move
a published faithfulness number. Duplication of a few dozen lines of graph traversal is a far
smaller cost than that coupling. The two modules also differ substantively: the detector
requires *exact* bipartiteness where the scorer reports a continuous purity, and the detector
carries witnesses so a property test can prove a reported cycle really closes.
**Consequences:** Two implementations of fan-width and cycle-finding exist and may drift.
That is accepted and is mitigated by each having its own tests; the drift that would matter —
a detector silently changed by a mining decision — is the one this prevents.

## D-032 — Salience lists are fixed now, before any narrative has been generated
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** `schemas/vocab_v1.yaml` carries, per typology, the field paths an adequate
narrative must mention, merged with a `_common` list of `focal_entity.id`,
`structure.n_nodes`, `structure.n_edges`. They are committed now, in Phase 3, and changing
one is a reviewed decision requiring its own entry here.
**Rationale:** Timing is the entire point. Salience decided after inspecting generation output
is not a standard, it is a description of whatever the model happened to produce, and every
"adequacy" number measured against it afterwards is circular. Fixing the lists before a single
narrative exists is what makes adequacy a test the system can fail. The lists are also part of
the Gold annotation protocol (Phase 6), so human annotators and the automated metric score
against the same definition rather than two that drifted apart.
Availability excuses omission: `required_fields` filters each list against the record before
anything is scored, so an Elliptic2 fan-out narrative is not penalised for omitting
`flow.total_outflow`, which no narrative on that substrate could faithfully mention. Without
that filter, adequacy would be systematically lower on the masked substrate for reasons that
have nothing to do with narrative quality, and the cross-substrate comparison would be
meaningless.
**Consequences:** A typology's list cannot be tuned after seeing results without a visible
diff and an entry here. `test_every_salience_path_is_a_real_field` prevents a list from
naming a field that does not exist, which would otherwise make a requirement permanently
unsatisfiable and invisibly depress the score.

## D-033 — Multi-currency aggregates are withheld, and the per-currency breakdown is always emitted
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** `flow.total_inflow`, `total_outflow`, `retained` and `max_single_transfer` are
`Money` objects only when every contributing transfer shares one currency; otherwise they are
`Unavailable("multi_currency_aggregate_undefined")`. `inflow_by_currency` and
`outflow_by_currency` are populated on **every** case regardless. Direction is measured in the
currency each side actually saw: an inflow uses `amount_received`/`receiving_currency`, an
outflow uses `amount_paid`/`payment_currency`. Near-threshold detection counts only transfers
denominated in `threshold_currency`.
**Rationale:** HI-Small carries fifteen currencies, 72,170 cross-currency transactions, and no
exchange rates. Summing 400,000 US Dollars and 3,000 Bitcoin produces 403,000 of nothing, and
a narrative quoting that total would be unfaithful in a way no checker could catch, because
the number would agree with the record. The same applies to a maximum: without a rate, 3
Bitcoin and 40,000 Rupee cannot be ordered.
The per-currency breakdown being unconditional is what makes the sentinel safe rather than
merely cautious. Nothing is lost except the sum that had no meaning — a generator can still
say "received EUR 200,000 and USD 300,000", which is both true and more informative than a
converted total would have been. Emitting the breakdown only in the multi-currency case would
have made the two branches structurally different and invited a consumer to special-case one.
Using each side's own currency matters because a cross-currency transfer has two different
amounts, and reusing one for both directions would invent a conversion silently. Restricting
near-threshold counting to the threshold's currency is the same principle: counting a 9,500
Rupee transfer against a 10,000 US Dollar threshold requires the rate the substrate lacks.
**Consequences:** A multi-currency case reports fewer aggregates, and the coverage report shows
it. `retained` additionally returns a sentinel when outflow exceeds inflow, which happens
legitimately because a case window is padded (D-019) and may catch an account dispersing funds
received before the window opened — reporting a negative "retained" would invite a narrative to
describe money that was never there.

## D-034 — The round-trip gate is paired with an independent oracle, because alone it cannot fail
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** The Phase 3 gate is two tests, not one. `test_round_trip_over_one_thousand_real_cases`
extracts facts, renders a probe narrative, checks every claim, and requires 100% SUPPORTED with
zero CONTRADICTED. `test_extractor_agrees_with_an_independent_oracle` recomputes fifteen
quantities directly from the raw Polars tables via `tests/oracle.py`, which imports nothing from
`g2t_aml.facts`, and compares.
**Rationale:** This was forced by mutation-testing the gate itself, and the result was not what
was expected. Three realistic extractor bugs were injected — `span_hours` returning seconds
instead of hours, an off-by-one in `structure.n_nodes`, and `in_degree` counting transactions
instead of distinct counterparties — and the round trip stayed at **100% SUPPORTED for all
three**. The reason is circularity: the probe renders its claims *from the fact record*, so a
wrong value is stated wrongly and then verified against itself. The round trip tests that the
extractor and the checker agree about semantics, which is a real and important property — it is
what stops the corpus and the metric drifting apart — but it says nothing about whether either is
correct.
The oracle closes it. Against the same three mutations it flags 148/150, 150/150 and 72/150 cases
respectively, and zero at baseline. It is deliberately naive: plain Python loops over
`to_list()`, the unit spelled out in `span_hours`'s name and the division left visible, no shared
helpers. Slower and written to be read.
An instrument that has only ever been compared against itself has not been calibrated, and
invariant 1 calls the fact layer a measurement instrument.
**Consequences:** Any quantity added to the fact record that a narrative will quote should gain
an oracle counterpart; the oracle currently covers structure, focal degrees, span, illicit
transaction count, single-currency totals and the currency inventory, and not the motif
descriptors, which are instead covered by the property-based witness tests (a detected cycle must
really close). `make facts-gate` runs both.

## D-035 — Ground truth wins outright, including when it says "no scheme"
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** On a substrate with `availability.typology_ground_truth`, `typology` is
resolved from the ground truth alone. A case carrying no stream label becomes
`{"label": "unclassified", "source": "ground_truth", "confidence": 1.0, "scope":
"stream_membership"}`. Motif inference runs **only** where there is no typology ground truth
at all — in practice, Elliptic2.
**Rationale:** The first implementation fell through to motif inference whenever
`view.typology` was None, which conflated two different situations: "this substrate cannot
tell us" and "this substrate tells us there is no stream". D-013 already draws that
distinction — `None` means no ground truth exists, `"unclassified"` means it exists and
reports no match — and the resolver was ignoring it.
**The cost was measured, not hypothesised.** The first full extraction over all 30,000 cases
produced this typology distribution: `bipartite` 7,337, `gather_scatter` 6,421, `fan_in`
2,817, `fan_out` 2,004 — against a Phase 2 ground-truth stratification of 357, 310, 269 and
289 respectively. Roughly 25,000 licit cases had been assigned a laundering typology purely
from their shape. That is not a rounding error, it is the majority of the corpus, and it
happens because *licit activity is structurally shaped like something*: an ordinary payroll
run is a fan-out, supplier settlement is a chain, collections are a fan-in. The AMLworld
authors say as much — fan-in and fan-out appear in both normal and alert categories because
criminals mimic legitimate activity, which is the entire premise of the hard-negative
population (`data/motifs.py`).
Left uncorrected, every Bronze narrative generated in Phase 4 from a licit case would have
asserted a laundering typology for a case the ground truth says is clean, and the generator
would have been trained on a corpus teaching it that structure alone implies a scheme —
precisely the failure the hard-negative population exists to prevent. The checker's guards
(an inferred typology claim returns UNVERIFIABLE, and `check_narrative_text` requires a
hedge) would have limited the damage at *evaluation* time while doing nothing about the
*training* corpus.
Nothing is lost by the fix. The structural finding stays in the `motifs` block, where it is a
shape rather than a scheme, with its full quantitative descriptors.
**How it was found:** by reading the coverage report from the first full run, not by a test.
The unit tests passed both before and after, because each fixture was checked against the
behaviour as written. The 20 golden files did catch it — 11 of them changed — which is what
golden files are for, but only once the decision had already been made to look.
**Consequences:** `typology.source == "inferred"` now appears only on substrates without
typology ground truth. Golden records were regenerated (11 of 20 changed) and
`test_licit_amlworld_case_is_ground_truth_unclassified_not_an_inferred_typology` locks the
behaviour. Phase 4 can rely on `source == "ground_truth"` meaning the label is authoritative
in both directions — positive and negative.

## D-036 — Typology is read from the case's own transactions, never from the seeding stream
**Date:** 2026-08-01 · **Phase:** 3 · **Status:** accepted
**Decision:** `_resolve_typology` computes the label from the dominant non-null `typology`
value on the case's own **edge table**, not from `CanonicalGraph.typology`. The record-level
invariant this establishes — **a typology other than `unclassified` always has
`labels.n_illicit_transactions > 0`** — is asserted in the unit suite and over the real
corpus.
**Rationale:** `CaseCollection.materialise` sets `CanonicalGraph.typology` from
`CaseRecord.typology`, which carries the typology of the stream the case was *seeded from*.
That is provenance about how the case was selected, not a fact about what it contains, and
the two come apart: measured on the built corpus, **346 of 30,000 cases (1.2%) carry a
stream typology while holding no laundering transaction at all**. The 48-hour window cap
(D-019) caught the seed account but none of its stream's flagged edges — Phase 2 itself
labels these `label="licit"` while `case_class="suspicious"`, so the disagreement is already
visible upstream and simply had no consumer until now.
Trusting the stream label would emit `{"label": "cycle", "source": "ground_truth",
"confidence": 1.0}` for a subgraph containing no cycle, no flagged transaction and no
evidence of any kind. `scope: stream_membership` warns a reader that a case may not exhibit
its typology *in full*; it does not license claiming one the case exhibits *not at all*, and
stretching it that far would make the scope field meaningless. A Bronze narrative built from
such a record would assert a scheme with nothing behind it, at full confidence.
Reading the edge column restores what `case_extraction._dominant_typology` already computes
at cut time and `materialise` then discards — so this is a correction back toward Phase 2's
own semantics rather than a new rule. The genuinely suspicious cases are unaffected: their
flagged edges carry the typology, so they resolve identically.
**How it was found:** by reading a single extracted record and noticing it claimed a `cycle`
typology alongside `n_illicit_transactions: 0` and `focal_is_illicit: false`. Every test
passed. The lesson is the same one D-035 taught an hour earlier — an extractor's failures are
visible in its *output*, not in assertions written against its own behaviour.
**Consequences:** Fixture cases must mark laundering on the *transactions*
(`tests.factories.as_laundering_stream`), not merely set a case-level attribute, because a
case-level attribute alone no longer produces a typology — the fixtures were under-specified
and are now representative of a real positive. Golden records were regenerated. Phase 4 can
rely on the invariant: if a fact record names a typology, the case contains flagged
transactions supporting it.

---

## D-037 — The training record embeds its fact record rather than pointing at one
**Date:** 2026-08-01 · **Phase:** 4 · **Status:** accepted
**Decision:** `training_record_v1.json` carries the complete `case_facts` record inline
under `facts`, `$ref`-ing the frozen fact schema rather than restating it. The same
document carries all three corpus tiers, differing only in `tier` and the `generator`
block.
**Rationale:** A narrative must be re-verifiable against *the record it was written from*,
not against whatever the fact store holds when someone re-runs the check. Those are the
same object today and need not be after a Phase 7 model-signal write-back, an extractor bug
fix, or any re-run of `make facts`. Verification that reads a *current* record rather than
the originating one is not verification, it is a coincidence that has held so far. Embedding
also makes a corpus file portable: Phase 5's verifier, Phase 9's trainer and Phase 10's
metric each need the facts, and a pointer makes all three depend on a directory none of them
own.
One schema for three tiers is the other half: a tier-specific schema would let Bronze,
Silver and Gold drift apart exactly where a reviewer wants them comparable, and would force
a migration the moment Silver landed. The ten-point harness gates all three with one
implementation, so "Silver is verified" means precisely what it means for Bronze.
**Consequences:** `bronze.jsonl` is 243 MB for 15,707 records, of which roughly two thirds
is `provenance` — the resolved fact config and the 139-entry field-producer map, repeated
per record. That is the price and it is paid knowingly: the file is gitignored build output,
it is read once per training run, and the alternative trades a storage cost for a
correctness risk. Changing `case_facts` still invalidates every corpus (invariant 9);
embedding does not change that, it makes it detectable, because check 3 reads the version
off the record rather than off the directory it came from.

## D-038 — Bronze excludes only single-account cases, not the whole degenerate tail
**Date:** 2026-08-01 · **Phase:** 4 · **Status:** accepted
**Decision:** A case enters the Bronze corpus when it is in the frozen Phase 2 split
manifest and has at least two accounts. 448 of 16,156 manifest cases are excluded, leaving
15,708. Two-account cases are kept and get their own `minimal_activity` family.
**Rationale:** Phase 3 deliberately deferred this ("Degenerate cases were NOT filtered...
It is a *corpus* decision, and it belongs to Phase 4"). Phase 2 flagged 18.1% of cases as
holding fewer than three accounts and the tempting move is to drop all of them.
That would be wrong twice over. A two-account case describes a **real transfer** with a real
amount, a real time and a real counterparty label — there is a report to write, and a corpus
that omitted them would be a corpus of only the structurally interesting cases, which is not
the population a deployed system sees. It would also delete 2,474 records, 16% of the
corpus, and skew the typology distribution toward the suspicious.
A **single-account** case is different in kind rather than degree: no counterparty is in
scope at all, `role` is `terminal`, every motif descriptor is null, and the only transactions
are self-loops. There is no activity between parties to describe, so any narrative is a
caption padded to the 80-token floor — which would inflate the surface metrics Phase 10
compares against Gold without containing a report.
The line is therefore drawn at "does this case contain an interaction", not at "is this case
interesting".
**Consequences:** `corpus.min_nodes` is a config value, so the decision is visible in every
run's resolved config rather than buried in a filter. `minimal_activity` needs its own family
because the structural families would render vacuously; it is 2,474 records, the second
largest family, and its narratives state the absence of structure positively rather than
falling silent. The excluded 448 are reported by the build, not dropped quietly.

## D-039 — The length gate is measured by a named, pluggable token counter that never falls back
**Date:** 2026-08-01 · **Phase:** 4 · **Status:** accepted
**Decision:** Validation check 6 uses a `TokenCounter` chosen by `corpus.tokenizer`. The
default, `heuristic-bpe-v1`, is a dependency-free over-approximation of Llama's BPE. Naming
a Hugging Face model or a local directory uses the real tokenizer and **raises** if it
cannot be loaded. The counter's identifier is written into every record's
`length.tokenizer`.
**Rationale:** The tokens that matter are Llama-3.1's, because that is what the fine-tune
consumes. But Phases 1–6 are CPU-only by decree (CLAUDE.md §4), `transformers` lives behind
the `llm` extra which pins `torch`, and Llama-3.1 is a gated download. Requiring it would
make `make bronze` impossible on the machine that builds every other CPU phase.
The heuristic leans **high** on purpose: a narrative passing [80, 400] under it passes under
the real tokenizer too. The asymmetry is chosen because the failure it prevents — a training
example silently exceeding the sequence budget — costs a training run, and the failure it
causes — a slightly conservative corpus — costs a few records at the margin.
The no-fallback rule is the part that matters. A run that asked for Llama, failed to load
it, and quietly measured with the heuristic would publish a length distribution attributed
to the wrong tokenizer, and nothing downstream could tell. Recording the counter's name on
every record makes a corpus re-gateable under a different counter without guessing what
produced the first result.
**Consequences:** Phase 9 should re-measure the corpus under the real tokenizer once the
GPU environment exists, and the recorded `length.tokenizer` is what makes that a comparison
rather than a replacement. The published length distribution must always be quoted with its
counter.

## D-040 — Claims are parsed out of the rendered text, never read from the fact record
**Date:** 2026-08-01 · **Phase:** 4 · **Status:** accepted
**Decision:** Every slot annotation carries both `rendered_value` (the string in the
narrative) and `raw_value` (what the record held). `g2t_aml.corpus.claims` builds the
checkable claim by **parsing `rendered_value`**, and `raw_value` is kept for diagnostics
only. Every formatter in `corpus/bronze/format.py` therefore ships with its inverse, and
each pair is property-tested for round-trip fidelity inside the checker's tolerance.
**Rationale:** This is D-034 applied to generation. Phase 3 found that the round-trip gate
could not fail, because the probe rendered its claims *from the record it was verifying
against* — three injected extractor bugs left it at 100% SUPPORTED. Bronze has exactly the
same trap available: a generator that emitted `Claim(value=facts.structure.n_nodes)` would
compare the record with itself and report every corpus ever built as perfectly faithful.
Parsing the text closes it. If `format_money` dropped a thousands factor, if a duration were
written in hours and read as days, if a display map were not a bijection — each produces a
claim that disagrees with the record, and the case comes back CONTRADICTED. Bronze's 100%
is then a measurement rather than a tautology, and the same machinery verifies Silver, whose
text nobody in this repository wrote.
**Consequences:** Every surface form must be invertible, which constrains the prose: role
names, typology labels and phase orderings go through display maps asserted to be
bijections, and a formatter that cannot parse its own output raises at render time rather
than producing an uncheckable span. `tests/unit/test_corpus_support.py` includes a slot
whose `raw_value` deliberately disagrees with its text, and asserts the claim follows the
text.

## D-041 — `graph_ref` resolution checks the graph's size against the fact record
**Date:** 2026-08-01 · **Phase:** 4 · **Status:** accepted
**Decision:** `graph_ref` is `<repo-relative case store>#<case_id>`. Validation check 2
resolves it against the Phase 2 membership tables and asserts the referenced subgraph has
exactly the node and edge counts `facts.structure` reports.
**Rationale:** A file-existence check catches the failure nobody has (a missing file) and
misses the one that matters: a reference that resolves to the *wrong* subgraph. That record
is trainable and silently wrong — the encoder sees one graph while the narrative describes
another, the loss still falls, and the faithfulness metric reads the embedded facts rather
than the graph, so nothing anywhere notices. It would surface only as a model that never
learned to condition on structure, months later, as a null result nobody could explain.
The two counts come from genuinely different places: the Phase 2 membership Parquet and the
Phase 3 extractor. Agreement between them is evidence, not a tautology. Measured over all
30,000 built cases, they agree exactly.
The reference names the case *store* rather than a per-case file because Phase 2 writes
membership as two columnar tables, and materialising 30,000 files to satisfy a reference
format would be cost with no benefit.
**Consequences:** The resolver reads each store's membership once and caches it, so the
check is a dictionary lookup per record rather than a Parquet scan. Phase 7 will want real
`.pt` tensors; the reference format already carries a store path and a case id, so pointing
it at a tensor store is a change to the resolver, not to the schema.

## D-042 — The four SAR sections are composed independently, not locked to one variant index
**Date:** 2026-08-01 · **Phase:** 4 · **Status:** accepted
**Decision:** A family's realisation index is a mixed-radix encoding of four independent
choices — subject, activity, pattern, basis — so each family offers 6 × 6 × 5 × 6 = 1,080
distinct narratives rather than five.
**Rationale:** The first design locked them together: variant *i* meant subject *i*,
activity *i*, pattern *i*, basis *i*. It gave five realisations per family, and measured
corpus self-BLEU of **0.811**. Since the four sections are grammatically independent —
nothing in the subject paragraph agrees with anything in the basis paragraph — composing
them independently costs nothing in authoring and multiplies the count by 216.
Worth recording honestly: **it barely moved self-BLEU** (0.811 → 0.810 on the same 584-case
sample), which is what prompted the investigation in D-043. It was still the right change,
on the evidence that actually bears on the question: distinct scaffolding skeletons went
from a handful per family to 12,324 across the corpus, and `no_finding` alone now uses all
1,080 of its realisations rather than 5.
**Consequences:** `generator.variant` is a realisation index, not a small ordinal, and is
only interpretable against the family that produced it — `Family.realisation` decodes it.
The acceptance criterion "4–6 realisation variants" is read as the *structural* count
(`Family.n_surface_variants`, the pattern-section count, which is 5 for every family); the
narrative count is reported separately because it is the one that governs diversity.

## D-043 — Self-BLEU is reported at a fixed five references, with the saturation curve published
**Date:** 2026-08-01 · **Phase:** 4 · **Status:** accepted
**Decision:** `measure_diversity` reports self-BLEU at five references per candidate, and
also publishes the whole curve at {1, 3, 5, 10, 50}. A second, non-saturating measure —
**distinct scaffolding skeletons**, the narrative with every slot span blanked — is reported
alongside it and is the number the collapse check actually keys on.
**Rationale:** The first Bronze build reported self-BLEU 0.81, which reads as a collapsed
template pack and would have triggered the brief's "add realisation variants before
proceeding". Multiplying the realisations by 216 (D-042) moved it by 0.001, which is what
made it worth measuring rather than reacting to. The curve on the *unchanged* corpus:

| references | 1 | 3 | 5 | 10 | 50 |
|---|---|---|---|---|---|
| self-BLEU | 0.16 | 0.36 | 0.48 | 0.63 | 0.82 |

Nothing about the corpus differs between those readings. With fifty references drawn from a
corpus written over a deliberately controlled vocabulary, almost every 4-gram of any
candidate appears in *some* reference and clipped precision goes to one. The metric was
measuring its own reference count.
Reporting 0.81 would have said the pack had collapsed when the pairwise number (0.16) and
the skeleton count (12,324 distinct forms for 15,707 records, 78%) both say plainly that it
has not. Reporting five references *without* the curve would be choosing the number that
flatters us. Publishing both is the only version of this that a reviewer can check.
Skeleton diversity is the better instrument for a *template* corpus and is introduced here
for that reason: it has no reference sample, does not saturate, and answers the actual
question — do two narratives differ only in the values they report — directly.
**Consequences:** Any self-BLEU quoted from this project must carry its reference count.
Phase 10 recomputes surface metrics with `sacrebleu` for the paper; this measure is a
build-time gate and the two must not be conflated. The `corpus.diversity.self_bleu_warn_above`
threshold of 0.60 is calibrated against the five-reference number and means nothing against
any other.

---

## D-044 — Silver is verified synthetic supervision, and the framing is enforced in code

**Date:** 2026-08-02 · **Phase:** 5 · **Status:** accepted
**Decision:** Silver is described, in the paper and in the code, as **verified synthetic
supervision** and never as distillation. Three mechanisms carry the claim, and each is a
hard failure rather than a convention:

1. **Two teachers from different families.** `specs_from_config` raises on fewer than two
   teachers and on two teachers of one family. It is not a warning and there is no override.
2. **Verification against the fact record, never against the other teacher.** A record is
   accepted because `facts/checkers.py` — the instrument that produces the paper's
   faithfulness numbers — finds zero contradictions and an unverifiable rate inside the
   published budget. Nothing in the pipeline reads one teacher's output while judging the
   other's, and there is no agreement or ensemble step anywhere.
3. **Evaluation against human-authored Gold**, in Phase 10. Silver is training supervision
   and is never a reference for a surface metric.

**Rationale:** If references are generated with a frontier model and an 8B student is then
scored on overlap against those same references, the number measures how well the student
imitates the frontier model. A competent reviewer will say exactly that, and they will be
right. The defence cannot be a paragraph in the paper asserting good intent; it has to be
a property of the artifact that a reader can check from the corpus file. Every Silver
record therefore carries its teacher, family, model, prompt hash and verification block, so
the two-teacher balance and the verification claim are both auditable without rerunning
anything.
The single-teacher refusal is deliberately placed at config load, before any money is
spent: a corpus discovered to be single-teacher after a six-hour run is a corpus that has
to be thrown away, and "it was cheaper" is not a defence a reviewer accepts.
**Consequences:** A Silver run needs credentials for two providers. `make silver` cannot be
run with one teacher configured, by design. Phase 10 must not report any overlap metric of
a Silver-trained model against Silver references; the Gold set is the only reference tier.

---

## D-045 — Sampling parameters are a per-teacher capability, not a global constant

**Date:** 2026-08-02 · **Phase:** 5 · **Status:** accepted
**Decision:** `TeacherSpec.supports_sampling` governs whether `temperature` and `top_p` are
sent. When it is False the fields are omitted from the request and recorded as **null on
the record, with a reason**, and surface diversity comes instead from a deterministic
per-case style directive drawn by a hash of the `case_id`. The phase brief's `temperature
0.7 / top_p 0.95` is honoured exactly for the open-weights teacher, which accepts them.
**Rationale:** The brief specifies temperature 0.7 and top_p 0.95 for diversity. Every
current frontier Anthropic model — Opus 5, Sonnet 5, Opus 4.8, Opus 4.7 — **rejects both
parameters with a 400**. They are not ignored and they do not degrade gracefully; a run
that sent them would fail on its first call and on all 12,000 after it. So the brief's
instruction and the available frontier models cannot both be satisfied literally, and there
were three ways out:

- Pick an older frontier model that still accepts sampling parameters. Rejected: choosing
  the teacher to fit a decoding knob is the wrong way round, and it dates the corpus.
- Send the parameters anyway and let the provider decide. Not available — it is a 400.
- Make the capability explicit per teacher and get diversity elsewhere for the models that
  cannot take it. Adopted.

The replacement mechanism matters as much as the decision. Surface variety for the frontier
teacher comes from eight style directives selected by `sha256(case_id)`, which is
**reproducible, auditable and recorded** — the rendered prompt hash on every record covers
the directive that produced it — where a temperature draw is none of those things. Depth is
controlled with `effort` instead, which is the parameter that replaced sampling on these
models.
Recording null *with a reason* rather than omitting the field is what keeps this honest: a
record that simply lacked a `temperature` key would be indistinguishable from one written
before the field existed, and a reader six months from now cannot tell "we did not set it"
from "the model refuses it".
**Consequences:** `temperature`, `top_p` and `provider_seed` are legitimately null on every
frontier-teacher record, and any analysis over the corpus must treat them as such rather
than as missing data. Adding a teacher requires stating its capability correctly; setting
`supports_sampling: true` on a model that rejects them fails the entire run at the first
call. The provider seed is recorded as `provider_seed`, never as `seed` — `seed` on a
training record is the run's global seed (invariant 5) and is an integer by frozen schema.

---

## D-046 — Repair is bounded at two attempts, and the discard log is a deliverable

**Date:** 2026-08-02 · **Phase:** 5 · **Status:** accepted
**Decision:** A rewrite that fails verification is repaired at most **twice**, then the case
is discarded and written to `silver_discards.jsonl` with its teacher, model, typology,
split, attempt count, per-class (H1–H9) violation breakdown, the checker's summary and the
verbatim spans the model invented. Verification discards and API-failure discards are
tagged with different `stage` values and reported as separate rates.
**Rationale:** This changes the standing instruction in CLAUDE.md §3, which said a rewrite
asserting an unsupported fact is "rejected, not repaired". Bounded repair is strictly better
than pure rejection on the evidence available: most first-pass failures are a single
unaligned figure, one targeted repair prompt recovers the record, and rejecting it outright
throws away a usable narrative and the money already spent generating it. What made pure
rejection attractive was the fear of an unbounded loop, and a hard limit of two addresses
that directly. CLAUDE.md has been updated rather than worked around.
Two, specifically, because past the second attempt a model stops repairing and starts
contorting: it satisfies the checker by deleting the sentence that carried the finding, and
the result passes verification while reading like nothing a human would file. An unbounded
loop also spends the budget precisely on the cases least likely to yield a usable record.
The discard log is an output because the number it produces is a genuine result. "A frontier
model handed a complete structured fact record *and* a correct draft still produced an
unrepairable factual violation in X% of cases" is the direct motivation for a
graph-conditioned architecture with a verifier in the loop, and it is invisible unless the
failures are instrumented on purpose rather than counted. Separating the API-failure stage
is what stops an afternoon of 503s from being reported as a property of a model.
**Consequences:** `silver_discards.jsonl` is a committed artifact of the run and a table in
the paper. The 15% ceiling in `corpus.verification.max_discard_rate` fails the build; it is
a gate on the run, not a target to tune the thresholds toward. If it trips, the finding is
the discard rate and the response is to report it, not to raise the budget.

---

## D-047 — Teacher assignment is stratified round-robin with a per-stratum offset

**Date:** 2026-08-02 · **Phase:** 5 · **Status:** accepted
**Decision:** Cases are assigned to teachers by ordering each `(typology, split)` stratum by
`sha256(case_id)` and round-robining down that ordering, starting at an offset drawn from
`sha256(typology|split)`. Not `hash(case_id) mod 2`.
**Rationale:** The brief proposes hash-mod-two, which balances *in expectation over the
corpus* and not within a stratum. This corpus has strata that are small — `stack` has 60
records, `fan_in` 70 — and a 60-case stratum split by a coin flip lands 40/20 often enough
to matter. "The open-weights teacher wrote most of the stack cases" is precisely the
confound the two-teacher design exists to remove, and it would be discovered after the run
rather than prevented.
Stratified round-robin balances every stratum to within one case, deterministically, on any
machine, with no seed — so the assignment is reproducible from the case manifest alone,
exactly as Bronze's variant selection is (D-042).
The per-stratum offset was added after the integration tests caught the naive version:
starting every stratum at teacher zero balances the large strata correctly and hands **every
singleton stratum to the same teacher**, which on a corpus with many rare typology/split
combinations reproduces the exact skew the stratification was introduced to remove. A
12-case fixture over 12 singleton strata came back 12/0.
**Consequences:** The balance report covers both the assignment *and* the corpus surviving
verification and filtering, with per-teacher retention rates and their spread — because a
teacher whose outputs are disproportionately discarded leaves a skewed corpus behind however
even the assignment was, and that asymmetry is itself a reportable finding.

---

## D-048 — Silver's claim extractor is slot alignment, and an unaligned quantity is a claim

**Date:** 2026-08-02 · **Phase:** 5 · **Status:** accepted
**Decision:** Silver's verification loop uses the fast deterministic extractor: Bronze slot
values located in the rewrite become claims **parsed from the rewrite's own text at the span
where they were found**, and every quantity, account identifier or risk descriptor that
aligns to nothing becomes a claim naming no fact field — which the checker resolves to
UNVERIFIABLE. Matching is exact; the teacher's output is canonicalised once beforehand.
Phase 10 adds an LLM extractor against the same `ClaimExtractor` protocol as a cross-check.
**Rationale:** The rewrite is a paraphrase of a text whose every load-bearing value is
already aligned to a fact field, so alignment is both cheap and mostly sufficient — and it
runs inside a repair loop thousands of times, where an LLM extractor would double the cost
of the tier.
The load-bearing detail is what happens to what does *not* align. Emitting nothing for an
unaligned figure would mean an invented number **raises** the supported rate, because the
denominator would only ever contain claims that were carried over correctly. Emitting a
field-less claim instead makes the checker return UNVERIFIABLE, which is the correct
three-valued answer — the figure has not been shown to be wrong, it has been shown to be
unbacked — and enough of them exhaust the 0.05 budget and trigger repair or discard.
Matching is exact rather than tolerant because every loosening is a place a real numeric
error gets absorbed: a matcher clever enough to see "USD 26,780" and "26,779.82 US Dollar"
as one claim is clever enough to wave through a genuine mistake. The conservative failure —
a correct-but-reworded value scored as unaligned — costs unverifiable budget and appears in
the discard log where a human can see it. The permissive failure is silent.
Exact matching is only safe because output is canonicalised first: a model that hard-wraps a
paragraph puts a newline inside a rendered value, and without canonicalisation that correct
figure would be scored as both a dropped fact and an invented one. Canonicalising the stored
text rather than loosening the matcher is what keeps `narrative[span] == rendered_value`
true on every record, which the harness asserts and Phase 10's evaluation depends on.
**Consequences:** Slot annotations on a Silver record are re-derived against the rewrite's
offsets, and a slot the rewrite dropped is **not** carried over — it shows up as reduced
salience coverage instead of as a mis-aligned span. Phase 10 can run both extractors over
the same corpus and compare, which is the cross-check this interface was shaped for.

---

## D-049 — Salience retention is a third acceptance condition, because omission is invisible

**Date:** 2026-08-02 · **Phase:** 5 · **Status:** accepted
**Decision:** A rewrite is accepted only when it has zero contradictions, an unverifiable
rate at or below 0.05, **and** still carries at least 95% of its typology's salient fields.
The third condition is the `min_fact_recall` the Silver config has declared since Phase 0,
enforced rather than documented.
**Rationale:** The brief's loop condition is contradictions and unverifiable rate. Both are
assertion-based, and neither can see the failure mode that matters most for a rewrite: a
model that drops the findings. "The subject account was reviewed. The activity warrants
further review." has zero contradicted claims, zero unverifiable claims, a perfect
unverifiable rate, and says nothing. It is fluent, faithful and useless, and it would be
written to the corpus and trained on.
H9 — omission of a material fact — is the one hallucination class in the taxonomy detected
by *absence* rather than by assertion, which is exactly why the other two conditions cannot
reach it. Adding a retention floor is a strictly tighter gate, not a relaxation, and so does
not conflict with the standing instruction never to loosen a threshold to reduce discards.
**Consequences:** A dropped salient field becomes a violation in the repair prompt, phrased
as "this fact is in the record and in the draft and your rewrite does not state it", so the
first repair attempt usually recovers it. `salience_coverage` is on every record and the
threshold is on the verdict that judged it, so no verdict can be re-read under a threshold
it was not produced under.

---

## D-050 — The system prompt is case-invariant, and a violation fails at load time

**Date:** 2026-08-02 · **Phase:** 5 · **Status:** accepted
**Decision:** The prompt files are split so the system message holds only what is identical
for every case in the corpus (role, controlled vocabulary, SAR structure, rewrite rules) and
the user message holds everything per-case. `assert_system_is_case_invariant` refuses at load
time any prompt whose system section uses a per-case placeholder, and the system message
carries a `cache_control` breakpoint.
**Rationale:** Over ~12,000 calls per teacher, a stable ~900-token system prefix is a
prompt-cache read on all but the first call instead of full-price input every time. That is
a material fraction of a $190–620 budget for a change that costs nothing but where the text
is placed.
It is enforced at load time because the failure is silent. A per-case value drifting up into
the system message makes every request a cache *write*; the generated narratives are
identical, the verification numbers are identical, and only the bill changes — and only in
aggregate, at the end of a long run. `system_prompt_hash` is recorded on every record so a
miss rate can be explained after the fact rather than guessed at.
**Consequences:** Editing a prompt file means keeping per-case values below the `<<<USER>>>`
marker. The marker parser was tightened to require a marker alone on its own line after the
first version split the file at a header comment that *named* the markers, producing a
141-character system message with the entire instruction block moved into the per-case half
— a bug with no visible symptom other than cost.

---

## D-051 — Hydra entrypoints capture their exit code, because `@hydra.main` discards it

**Date:** 2026-08-02 · **Phase:** 5 · **Status:** accepted
**Decision:** Every pipeline script keeps its Hydra-decorated entrypoint as `_run`, records
its status in a module-level `_EXIT_CODE`, and exposes a thin `main()` that returns it.
`tests/integration/test_repo_contract.py` asserts the shape on every script carrying a
`@hydra.main`.
**Rationale:** `@hydra.main` **returns None regardless of what the wrapped function
returns.** Every script in this repository ended with `sys.exit(main())` over a decorated
`main`, so every one of them exited 0 unconditionally — including on the paths that
carefully `return 1`.

This was live in Phases 1 through 4 and was found by accident in Phase 5, when a Silver
preflight failure printed its errors and then exited 0. The consequence is worse than the
cause: `make bronze` on a failed ten-point gate exited 0, `make smoke` could not fail, and
CI would have gone green on a corpus that failed its own gate. Four phase logs record gates
as "passed" and are not wrong — the gates were checked and reported in their output — but
nothing downstream of those scripts could have told a pass from a failure, so the
enforcement described in CLAUDE.md's invariant table was not actually in place for any
`return 1` path.

The fix keeps the Hydra entrypoint shape rather than restructuring eight scripts, and the
contract test asserts the shape rather than the behaviour because running eight pipeline
stages inside a unit test is not a unit test.
**Consequences:** `04_build_bronze.py` now exits 1 on a schema mismatch and on a gate
failure, verified. Any new pipeline script must follow the same shape or the contract test
fails. This is the second time a repo-wide contract test has caught something review would
not have (the first was the hardcoded-path grep in Phase 4), and both were cheap.

---

## D-052 — The hard-negative floor beats typology balance, because on this data they cannot both hold

**Date:** 2026-08-03 · **Phase:** 6 · **Status:** accepted
**Decision:** The Gold sample is allocated as three named blocks — a hard-negative block
(28%), a typed block balanced evenly across the eight typologies under capacity (44%), and
an ordinary-unclassified block (28%) — rather than as an even nine-way split over
`typology.label`. The `hard_negative_floor` (0.25) is asserted **after** allocation and
refuses the sample rather than adjusting it.
**Rationale:** The phase brief asks for two things that are jointly unsatisfiable on this
population, and the conflict is a fact about the data rather than a modelling choice.
*Every* hard negative is licit, so it carries no laundering stream, so its `typology.label`
is `unclassified` — 839 of 839 in the AMLworld test split. An even nine-way allocation
therefore caps `unclassified` at about a ninth of the budget, and a 25% hard-negative floor
is unreachable inside it.

The floor wins because hard negatives are the stratum whose absence is unrecoverable. They
are legitimate activity whose *shape* looks like laundering, and a Gold set without them
cannot demonstrate restraint being exercised — which is the behaviour the brief singles out
as where systems fail and where a human reference is worth the most. Typology balance
degrades gracefully by comparison: eighteen `stack` narratives instead of twenty-two is a
smaller loss than zero evidence that a human declined to escalate a payroll run.

Naming the three shares in the config rather than deriving them keeps the trade-off
visible. A derived allocation would encode this collision as an accident nobody wrote down,
and the next person to change `n_cases` would silently change the balance.
**Consequences:** The achieved sample is 350 cases at 28.3% hard negatives, with the eight
typed typologies at 18–20 each (spread ≤ 2) and all three size buckets populated. `stack`
and `scatter_gather` sit at 18 because the test split holds only 19 and 56 of them
respectively; `allocate_evenly` (promoted to a public name in `case_sampling`, shared with
Phase 2) hands the remainder back rather than shrinking the sample. Changing the shares is
a config edit; dropping the floor below 0.25 is refused at construction.

---

## D-053 — Elliptic2's unmet quota is reallocated, and the reallocation is reported as a deficit

**Date:** 2026-08-03 · **Phase:** 6 · **Status:** accepted
**Decision:** The sampler carries a `reallocate_deficit` flag, default true. A substrate
that cannot fill its share hands it to the substrates that can, and **both** the original
deficit and the reallocation are recorded in `gold_sample.json` and in the reservation's
provenance.
**Rationale:** The brief specifies roughly 70% AMLworld / 30% Elliptic2. Elliptic2 access
has still not been requested (open since Phase 1), so its 105-case quota is unobtainable.
Two bad options and one acceptable one:

*Silently reallocate* — the sample reaches 350 and nothing records that it covers one
substrate. That is the option that puts a wrong claim in the paper.

*Refuse to reallocate* — the sample is 245 cases, below the 300–400 target, and 105 items
of annotation capacity are spent on nothing while the deficit is recorded. Honest but
wasteful, and it makes the Gold set smaller than the phase needs for no benefit.

*Reallocate and report* — 350 cases, with `dataset:elliptic2: 105 requested, 0 supplied`
and `reallocated: 105 requested, 105 supplied` in the report. The sample reaches its target
and cannot be mistaken for one that met the substrate split.

The deficit is never absorbed into the allocation arithmetic, which is the property that
makes the third option different from the first.
**Consequences:** The Gold set is AMLworld-only until access is granted, and the
recruitment brief's paper-facing text states that as a limitation rather than a footnote.
If Elliptic2 arrives before annotation closes, a second sample can be drawn for it and
appended; the reservation's hash is over the id set, so extending it is a visible change.

---

## D-054 — The fact panel renders values with Bronze's formatters, and Bronze is used nowhere else in the interface

**Date:** 2026-08-03 · **Phase:** 6 · **Status:** accepted
**Decision:** `human/factpanel.py` imports Bronze's *formatters* (`format_money`,
`format_count`, `format_duration`, `format_timestamp`, `format_density`, `format_percent`)
and its `ROLE_DISPLAY` map, and renders every value through them. It imports no template, no
renderer and no narrative. `tests/integration/test_repo_contract.py` asserts that no
annotator-facing module can reach `render_bronze`, `BronzeNarrative`, `target_narrative` or
the slot extractor.
**Rationale:** Found by writing a narrative by hand against the panel and watching it fail.
Ingestion aligns a Gold narrative against Bronze's slot values by **exact string match**
(D-048). The panel's first version rendered `9,434.82 Canadian Dollar` where Bronze renders
`9,435 Canadian Dollar`, so an annotator who copied the panel *correctly* produced a value
that aligned to nothing — scored as a dropped fact **and** as an invented quantity, on
every monetary case in the corpus. The same defect existed for entity roles: the controlled
vocabulary spells them `conduit account` while Bronze writes `a conduit account`, and
`focal_entity.role` is on the salience list for three typologies, so the panel would have
made that requirement permanently unmeetable.

Sharing the formatters is what makes "write down what the panel says" the behaviour the
pipeline rewards. The alternative — loosening the matcher — is the one D-048 already
rejected, and for the same reason: every tolerance a fuzzy matcher grants to a correct
paraphrase it also grants to a real error.
**Consequences:** A change to a Bronze formatter changes what an annotator sees, which is
correct: the two must agree by construction. The contract test draws the line between a
formatter (permitted) and a narrative (forbidden), so the exception cannot widen quietly.
The same investigation fixed the threshold row, which now shows the vocabulary's whitelisted
citation phrase — the panel previously taught `the 10,000 US Dollar reporting threshold`,
which is H6 because the whitelist reads `the USD 10,000 reporting threshold`.

---

## D-055 — Live validation flags but never blocks, and the overrides are a deliverable

**Date:** 2026-08-03 · **Phase:** 6 · **Status:** accepted
**Decision:** The interface flags forbidden phrases, out-of-inventory accounts,
masked-substrate assertions and length violations as the annotator types, and lets them
submit anyway. Every flag is stored with the text that raised it and whether it was
overridden, and `gold_quality.json` reports override rates per rule. Submission is gated on
a *different* check: the Phase 3 checker is run over the draft and any CONTRADICTED verdict
is shown before the item can be saved.
**Rationale:** A blocking validator trains people to write around the checker. The phrase
gets rephrased until the highlight disappears, and what the corpus then measures is the
annotator's skill at evading a regex rather than their accuracy. It also destroys the
phase's most useful by-product: a rule overridden by two calibrated domain experts on a
fifth of items is evidence the *rule* is wrong, and the only way to discover that is to let
the writing through and count.

The distinction that makes this safe is that the two checks are not the same kind of thing.
The live flags are a word list, which a person can out-argue. The submit-time check compares
against the fact record, and the only way to clear it is to state what the record says.
**Consequences:** A Gold narrative can enter the corpus carrying an overridden critical
flag, and that is intended — it will be visible in the quality report and in the second
review. Phase 6 found one such gap already: the frozen vocabulary's guilt list carries
`is money laundering` and `is guilty of` but **not** the equally natural `is laundering
money`, so that phrasing passes the text scan. It is recorded rather than patched, because
editing a frozen vocabulary to make a test pass is what makes a frozen artifact meaningless.
Whether to bump `vocab_v1` for it is a Phase 10 decision with a corpus regeneration attached.

---

## D-056 — Gold reuses `generator.renderer_version` rather than bumping the frozen record schema

**Date:** 2026-08-03 · **Phase:** 6 · **Status:** accepted
**Decision:** A Gold training record sets `generator.renderer_version` to
`GOLD_INGEST_VERSION` — the version of the ingestion pipeline and the annotation protocol
it enforces — alongside `generator.method: "human"`, `annotator_id`, `reviewer_id`,
`adjudication` and `protocol`.
**Rationale:** `training_record_v1` is FROZEN and *requires* `generator.renderer_version` on
every record, documented as "version of the code that produced the narrative, so a corpus
can be attributed to an exact renderer". For Gold the narrative was produced by a person.
Two options: bump the schema to make the field optional, or read the requirement honestly.

Bumping it would invalidate every Bronze and Silver record on disk (invariant 9) to
accommodate one tier's vocabulary — 15,707 records regenerated so that one field could be
absent from 350. The honest reading is available and costs nothing: for Gold, the pipeline
*is* the thing that produced the record, and it is versioned. `generator.protocol` names the
guidelines document and `generator.annotator_id` names the author, so nothing is obscured.
**Consequences:** Found by the ten-point harness on the first hand-written narrative, which
is the harness doing its job across a tier it had never seen. A reader of `gold.jsonl` sees
`method: human` beside `renderer_version: 1.0.0` and must know to read the latter as the
ingestion version; the field's meaning is documented at `GOLD_INGEST_VERSION` and here.

---

## D-057 — Gold's unverifiable budget is enforced at ingestion, because the harness cannot see an unaligned quantity

**Date:** 2026-08-03 · **Phase:** 6 · **Status:** accepted
**Decision:** `ingest_annotations` holds any record whose *extractor* unverifiable rate
exceeds `MAX_UNVERIFIABLE_RATE`, in addition to the ten-point harness that runs afterwards.
**Rationale:** The harness rebuilds a record's claims from its `target_slots`
(`claims_from_slots`). For Bronze that is complete, because every value in the text came
from a slot. For Gold — and for Silver — the slots are exactly the values that **did**
align, so a quantity the annotator introduced that aligns to nothing produces no slot and is
therefore invisible to check 5. A Gold narrative could carry a dozen unbacked figures and
pass the gate by virtue of not having produced slots for any of them.

The extractor *does* see them: it emits a claim naming no field, which the checker resolves
to UNVERIFIABLE, and that rate is written onto the record. Enforcing it at ingestion is the
smallest correct fix. Changing `validate_corpus` to re-extract would give the harness a
dependency on the Bronze reference and would change what "the ten-point gate" means for
Phases 4 and 5 as well — a larger change, and one that belongs with Phase 10's extractor
comparison rather than here.
**Consequences:** A Gold record is gated twice, and the ingestion gate is the stricter of
the two. The first hand-written test narrative was held by it, correctly: it cited "the
10,000 US Dollar reporting threshold" in words the whitelist does not carry. The same
investigation added a regulatory-citation pass to the shared extractor — a whitelisted
citation is now a REGULATORY claim rather than an unaligned number, which was costing any
Silver rewrite that cited one 6% of its budget for a sentence the vocabulary explicitly
permits.

## D-058 — GATv2 rather than GAT, and all six arms are built now rather than later

**Date:** 2026-08-03 · **Phase:** 7 · **Status:** accepted
**Decision:** The primary encoder is **GATv2**; the original GAT is not implemented at all.
All six comparison arms — GATv2, GINE, GraphSAGE, GCN, a virtual-node graph transformer,
and an MLP control — are built in this phase behind one interface, not deferred to Phase 11.
**Rationale:** Two separate points.

On GATv2: the original GAT computes `a^T·LeakyReLU(W[h_i ‖ h_j])`, so the attention
vector is applied *after* the nonlinearity and the ranking of neighbours is **static** —
there exists a global ordering of keys that every query agrees on, regardless of the query.
GATv2 moves the nonlinearity inside, making the ranking query-dependent. Brody et al.
(2022) prove the first is a strict subset of the second, and the two cost the same. There
is no argument for the original other than inertia, so it is not offered as an option — an
arm nobody should choose is a way for someone to choose it by accident.

On building all six now: an ablation whose arms were written months apart differs in more
places than the one being measured. Building them together against a shared
`BaseEncoder` means input projection, edge encoding, attention pooling and both heads are
*literally the same code* in every arm, and `message_passing` is the only override. A
performance gap is then attributable to message passing rather than to one arm having got
a better edge encoder because it was written second.
**Consequences:** `configs/encoder/gat.yaml` is replaced by six per-arm configs and
`configs/config.yaml` defaults to `gatv2`; `test_hydra_compose` was updated to match.
`SAGEConv` and `GCNConv` have no edge-attribute channel, so those two arms genuinely
cannot see amounts or payment rails; that is reported as part of what the comparison
measures rather than papered over with a synthetic edge path. Phase 11 inherits the arms
and needs to write no new architectures.

## D-059 — Node features are case-local, and the interim table's global aggregates are excluded

**Date:** 2026-08-03 · **Phase:** 7 · **Status:** accepted
**Decision:** Every node feature is recomputed from the case's own edges. The interim node
table's `in_degree`, `out_degree`, `degree`, `total_received` and `total_sent` columns are
on the `PERMITTED_NODE_COLUMNS` deny-by-omission list and are never read.
**Rationale:** Those five columns are *global* aggregates over the whole 515,088-account
graph. CLAUDE.md note 8 already records what reading them does to the fact layer — it
makes every narrative unfaithful, because "received from twelve accounts" would mean
twelve across all of HI-Small rather than twelve in this case. The trap here is different
and worse. The splits are **temporal** (invariant 2), and a global aggregate is computed
over both sides of the boundary. A test-window account's global degree therefore encodes
activity from the training window and from after the test boundary, so the encoder could
read a case's future off its node table. Every aggregate metric would look fine.

A test asserts the exclusion behaviourally rather than by inspection: it overwrites all
five columns with absurd constants and requires the resulting tensor to be byte-identical.
**Consequences:** The feature builder is ~25 ms per case rather than a column read, which
is why the cache exists. The MLP control gets exactly the same case-local features — degree,
transaction counts, reciprocity, local clustering, amount aggregates, burst timing — which
is what makes it a hard baseline rather than a strawman, per the Phase 7 brief.

## D-060 — AUC-PR is the selection metric; ROC-AUC is reported and never selected on

**Date:** 2026-08-03 · **Phase:** 7 · **Status:** accepted
**Decision:** Early stopping, checkpoint selection, the bootstrap intervals and every
claim of superiority go through **average precision (AUC-PR)**. ROC-AUC is computed and
reported for every population and is used for nothing.
**Rationale:** ROC-AUC is computed against the false-positive *rate*, whose denominator is
the negative count. On the realistic-imbalance stream (9,270 negatives, 730 positives),
a thousand additional false positives move FPR by 0.11 and precision from 0.42 to 0.13.
A model can therefore look excellent on ROC while being useless to an investigator who has
to work the alert queue. The unit test `test_auc_roc_flatters_where_auc_pr_does_not` pins
the phenomenon on synthetic data so the claim is not merely asserted in prose.

ROC-AUC is still reported because the AML detection literature quotes it almost
universally, and omitting it would make this work incomparable to prior art.

AUC-PR's baseline is prevalence, not 0.5, so **prevalence and `lift = auc_pr / prevalence`
are reported beside every AUC-PR** — an AUC-PR quoted alone is not interpretable, and the
balanced test split (15.9% prevalence) and the realistic stream (7.3%) have different
floors.
**Consequences:** `training.early_stop_metric` is `val_auc_pr` and there is no code path
that selects on anything else. Selecting on one metric and reporting another is the
specific failure this forecloses.

## D-061 — Attention pooling to k tokens is the readout, and it is built in Phase 7 for Phase 8

**Date:** 2026-08-03 · **Phase:** 7 · **Status:** accepted
**Decision:** Every arm reads out through `AttentionPooling`, which produces
`[B, 16, 256]` query-attended graph tokens. The single `[B, 256]` graph embedding is the
mean over those tokens, not a separate pooling path.
**Rationale:** Mean-pooling a 150-node subgraph to one 256-vector is an information
bottleneck placed exactly where the structure lives, and the structure is the contribution.

The operative reason is Phase 8. The fusion layer projects a *sequence* of graph tokens
into the language model's embedding space, so `[B, k, d]` has to exist eventually. Building
the pooling head inside the encoder means Phase 8 consumes this output directly instead of
bolting a second pooling stage onto an already-trained encoder — which would be a pooling
head trained by the language-model loss alone, on a corpus far too small for it.
`configs/fusion/prefix.yaml` now interpolates `num_prefix_tokens` from
`encoder.n_pooled_tokens` so the two cannot silently disagree.

The softmax is scattered **within a graph**, never across the batch, which a unit test
asserts by requiring each graph's attention to sum to one per token. Pooling weights are
also the attribution written into `model_signal.top_contributing_nodes`, so the readout and
the interpretability story are the same mechanism rather than two that could disagree.
**Consequences:** Pooling costs k scatter-adds per forward pass; the fused `[N, k, d]`
form allocates 157 MB on a full batch and does not fit alongside the backward graph on a
4 GB card, so the loop is deliberate and commented. `encoder.out_dim` is gone — an arm's
output width *is* its `hidden_dim`, and carrying a second declared width was how the two
could drift.

## D-062 — Two positional encodings, with per-graph sign flipping on the Laplacian block

**Date:** 2026-08-03 · **Phase:** 7 · **Status:** accepted
**Decision:** Node features carry 8 Laplacian eigenvector components and 16 random-walk
return probabilities. During training the Laplacian block's signs are flipped at random,
**per graph**. The PE ablation zeroes the block rather than rebuilding the cache.
**Rationale:** Message-passing GNNs are bounded by 1-WL and provably cannot count or
detect cycles; `cycle`, `stack` and `fan_out` are literally structural motifs, so that
bound bites directly here. The two encodings are complementary: the Laplacian eigenbasis
separates communities and elongated chains but carries a sign ambiguity, while the
random-walk encoding is sign-unambiguous and its k-th entry is the return probability at
exactly k steps — a direct, local measurement of cycle structure at every length up to 16.

The sign flip is drawn per *graph*, not per batch. Each case's eigenbasis has its own
independent ambiguity, and one draw shared across a batch would teach the model that all
cases in a batch flip together — an artefact of batching rather than a property of the
encoding. This was a real bug in the first implementation.

Ablating by masking rather than by rebuilding keeps the input width, the parameter count
and the initialisation identical, so the comparison isolates the information rather than
the capacity.
**Consequences:** The Laplacian is a dense `eigh`, which is fine at n ≤ 150 and would not
be at graph scale; the encoder is a *case* encoder and this is the assumption that makes
it cheap. Isolated nodes get all-zero encodings — a walk from a node with no edges is
undefined rather than certain to return — and 41.5% of cases have fewer than five nodes,
so zero-padding the eigenbasis is the common path rather than an edge case.

## D-063 — `model_signal` is written back, and Bronze is deliberately **not** regenerated

**Date:** 2026-08-03 · **Phase:** 7 · **Status:** accepted
**Decision:** `scripts/07b_score_cases.py` populates `model_signal` in all fact records
and refreshes the aggregate Parquet's two score columns. The Bronze corpus is **not**
regenerated, and a test pins `gnn_risk_score=none` in `bronze.jsonl`.
**Rationale:** The regeneration question has two halves and they have different answers.

**Narratives: unaffected.** No Bronze template reads `model_signal`; the renderer never
touches the block. Not one of the 15,707 narratives changes, so the corpus, its validation
report and its diversity report all stand. Nothing about the frozen schema changes either —
the block has existed at 1.0.0 since Phase 3, null-valued, and `with_model_signal` was
designed for exactly this. It is *populated*, not added, so invariant 9 is not engaged.

**The serialisation baseline: at risk, and this is the finding.**
`facts.serialiser._compact` emits `gnn_risk_score`, and that string is stored verbatim as
`serialised_facts` on every training record. Regenerating Bronze after the write-back would
therefore push the encoder's own risk score into the input of the **serialisation
baseline** — the "flatten the facts, no graph encoder" ablation arm. That arm would then be
reading the encoder it exists to be compared against, and every "graph fusion beats
flattened facts" number in Phases 9 and 12 would be measuring the wrong thing. Nothing
would fail; the baseline would just quietly get better.

So Bronze is left alone, and the invariant is enforced rather than remembered:
`test_bronze_serialised_facts_carry_no_model_signal` reads the corpus and fails if any
record's `serialised_facts` carries a real score. `score_cases.json` records
`bronze_regeneration_required: false` with the reasoning, so the decision travels with the
artifact.
**Consequences:** The write-back makes the on-disk fact record diverge from the copy
**embedded** in every training record (D-037). That divergence is correct and is confined
to `model_signal`: the embedded copy is a snapshot of what the narrative was actually
written from, and the narrative was written from a record with no model signal.
`test_the_embedded_facts_are_the_record_the_narrative_was_written_from` now compares every
block *except* `model_signal` and still requires the rest to match byte for byte, so a real
drift between the corpus and the fact store is still caught. This test failed on the first
write-back run and is how the divergence was found.

Phase 9's serialisation baseline must keep consuming `bronze.jsonl` as committed. If a future phase genuinely needs a fact serialisation *with* the model signal —
for instance to test whether the LLM can use a scalar risk score — it must be a separately
named field, not a regeneration of this one. `score_percentile` ranks within the population
scored in a single run, and that population size is recorded in the report, because a
percentile without its reference population is the same class of unreadable number as
self-BLEU without its reference count (D-043).

## D-064 — GATv2 stays the primary arm although GIN and the graph transformer out-score it

**Date:** 2026-08-04 · **Phase:** 7 · **Status:** accepted
**Decision:** `gatv2` remains the primary encoder. `graph_transformer` and `gin` post
higher mean test AUC-PR and the primary arm does **not** change.
**Rationale:** The Phase 7 brief asked for this to be stated plainly if it happened, so:
over three seeds, `graph_transformer` reaches 0.8877 ± 0.0190 and `gin` 0.8801 ± 0.0056
against GATv2's 0.8720 ± 0.0136. GATv2 is third of six.

The decision is made on the paired bootstrap, not on those means. Neither gap survives it:

| `gatv2` minus | mean difference | per-seed | excludes zero at every seed |
|---|---:|---|---|
| `graph_transformer` | −0.0157 | −0.0182, −0.0213, −0.0077 | no |
| `gin` | −0.0082 | +0.0003, −0.0054, −0.0195 | no |

Against `mlp`, `sage` and `gcn` the same test excludes zero at every seed, so the
machinery is not simply incapable of detecting a difference — it detects three and
declines to call these two.

Given three statistically indistinguishable arms, the tiebreak is what each costs the rest
of the project. `graph_transformer` needs 1.63× the parameters and ran ~3× slower per
epoch (17 s against 5 s), and its attention is computed over a graph augmented with a
virtual node, so a layer-wise attention figure includes a node that does not correspond to
any account. `gin` has no message-passing attention at all. GATv2 has edge-conditioned
attention over exactly the real transactions, which is what the paper's interpretability
figures and the `model_signal.top_contributing_nodes` attributions are read off.

Two things this decision is **not** allowed to become. It is not a claim that GATv2 is the
best architecture — it is third, and the write-up says so. And it is not permanent: every
arm is built, checkpointed and evaluated, so Phase 11 can switch the primary by changing
one config key.
**Consequences:** Phase 8 fuses GATv2's pooled tokens and Phase 11 reports all six. **What
would change this:** a wider seed set, or a tuning budget per arm, that made either gap
exclude zero at every seed. Every arm ran at one configuration and none was tuned, so
these numbers compare architectures at a reasonable setting rather than at their best, and
that limitation belongs in the paper next to the table.

## D-065 — Focal loss is kept as the default and reported as no better than weighted BCE

**Date:** 2026-08-04 · **Phase:** 7 · **Status:** accepted
**Decision:** `training.loss` stays `focal(gamma=2.0)`, and the results report that
weighted BCE is indistinguishable from it — marginally ahead on the mean.
**Rationale:** Both were implemented and both were run at three seeds, because "we used
focal loss for the imbalance" is not a result unless the alternative was measured. Both
take the same inverse-frequency `alpha`, so the comparison isolates the focusing term
rather than confounding it with a different class balance; `FocalLoss(gamma=0)` and
`WeightedBCELoss` are asserted equal by a unit test, which is what makes that claim
checkable.

The paired difference is **−0.0056** in favour of weighted BCE, does not exclude zero, and
flips sign across seeds (+0.0139, −0.0199, −0.0108). On this corpus the focusing term
bought nothing over inverse-frequency weighting.

Focal is kept rather than switched because the two are statistically identical, it is what
the phase brief specified, and switching on a non-significant 0.006 would be selecting a
loss on noise — the same error the AUC-PR-versus-ROC discipline (D-060) exists to avoid.
**Consequences:** The paper must state the comparison, not just the choice. A reader who
sees "focal loss" and infers that the imbalance needed it would be drawing a conclusion
this phase's own evidence does not support.

## D-066 — Positional encodings are retained despite contributing nothing measurable

**Date:** 2026-08-04 · **Phase:** 7 · **Status:** accepted
**Decision:** The 8 Laplacian and 16 random-walk components stay in the feature space, and
the ablation result — **+0.0023, not significant, sign-flipping across seeds** — is
reported as a negative finding.
**Rationale:** D-062 included them on a real theoretical argument: 1-WL-bounded message
passing cannot count or detect cycles, and `cycle`, `stack` and `fan_out` are structural
motifs. The argument did not pay on this substrate, and the likely reason is in Phase 2's
own numbers — the median case has 6 nodes and the 90th percentile 30, so a three-layer
network already reaches essentially the whole graph and there is little for a positional
coordinate to add that message passing has not already computed.

They are kept for three reasons. They cost 24 of 51 input dimensions and no measurable
accuracy. Elliptic2's subgraphs are drawn from a 49-million-node graph and may be very
much larger, where the argument could hold and where removing the channel now would mean
rebuilding the cache and every checkpoint. And a retained-but-ablated component with a
published null result is more useful to a reader than a silently removed one.

**The paper must not claim they help.** A negative result kept and reported is invariant 7
working; a negative result quietly deleted, leaving a feature the reader assumes was
load-bearing, is the failure mode.
**Consequences:** `training.lap_pe_sign_flip` and the sign-flip augmentation stay in the
training loop for a component that does not measurably matter, which is a small cost paid
for keeping the option open. If Phase 11 confirms the null on Elliptic2 as well, Phase 12
should drop them and bump `FEATURE_SPEC_VERSION`.

## D-067 — Resume is guarded by the training regime, because a smoke checkpoint nearly became a result

**Date:** 2026-08-04 · **Phase:** 7 · **Status:** accepted
**Decision:** Checkpoints record their `training_config`, and `training.resume=true`
refuses any checkpoint whose `epochs`, `loss`, `lr` or `batch_size` differ from the active
config. A checkpoint with no `training_config` is refused outright.
**Rationale:** Resume exists because the first full sweep was OOM-killed 24 runs in and
retraining four hours of compute to recover three missing runs is indefensible. But a
checkpoint carries no evidence of how long it trained once its weights are loaded, so a
two-epoch wiring-check checkpoint left in the checkpoint directory resumes exactly as
happily as a converged one — and its number lands in a results table looking like every
other row.

This is not hypothetical. `artifacts/checkpoints/encoder/gatv2_bce/gatv2_seed42.pt` was a
two-epoch smoke artifact, and the first resume run would have published it. It was caught
by comparing file timestamps against the sweep's start time, which is not a control — it
is luck, and it does not survive the next person.

The four guarded keys are what a results table means by "trained the same way". `resume`,
`seeds` and the bootstrap settings are excluded deliberately: they say nothing about how
the weights were produced, and requiring them to match would refuse every legitimate
resume. Metrics are always **recomputed** from the restored weights rather than read back
out of the old result JSON, so a resumed row is a measurement rather than a copy of a file.

`epochs_run` and `seconds` are recorded as 0 on a resumed run. That is deliberate: the run
did no training, and carrying the original timings forward would put fabricated wall-clock
into the results.
**Consequences:** The 24 checkpoints that survived the OOM predate the field and would
have been refused, so they were stamped with the config the sweep log proves they ran
under (a single `experiment=encoder_sweep` run, 21:29–01:17, all four keys at their
defaults) rather than retrained. That backfill is recorded here and in PHASE_LOG because
it is the one place in this phase where a provenance field was written by hand rather than
by the run that produced the artifact.

---

## D-068 — Phase 9 was built but not trained: the machine cannot hold the model

**Date:** 2026-08-04 · **Phase:** 9 · **Status:** accepted
**Decision:** The Phase 9 harness, the Phase 8 fusion layer and the inference guard are
implemented, tested and configured. **No arm was trained.** S1, A1, S2, B7 and B8 are all
deferred, and Gate 8 — the project's decision point — remains open.
**Rationale:** Three preflight conditions failed, and none is recoverable by writing code.

| Condition | Required | Actual |
|---|---|---|
| GPU | ≥24 GB, 48 GB comfortable | **RTX 2050, 4 GB** (3.3 GB free), 7 GB system RAM |
| Silver corpus | ≥8,000 verified | **0** — Phase 5 machinery complete, no API credentials |
| Gold | held-out test set written | **0 narratives** — no annotator recruited |

Llama-3.1-8B at nf4 with double quantisation is ~4.5–5.6 GB of weights alone, more than
the card's total capacity before a single activation, LoRA parameter or optimiser state.
The usual escape hatch is CPU offload; with 7 GB of system RAM and 7 GB of swap already in
use, that is closed too. This is not the "reduce `max_seq_len` to 1536" situation the brief
anticipates — that guidance assumes a 24 GB card and buys back activation memory, not
weights. Phase 7's 27 encoder runs fit in 4 h 10 min on this card because the encoder is
628k parameters; the generator is four orders of magnitude larger.

The corpus blockers compound it. The configured curriculum is Bronze+Silver for epoch 1 and
Silver only for epochs 2–3; with Silver at zero, two of three epochs have no data. With
Gold unwritten there is no held-out human reference to evaluate against.
**Consequences:** Everything that does not depend on the missing inputs was built and is
verified on CPU against a stub backbone: 119 new tests, the overfit check, the loss-mask
assertions, the fp32-projector assertion, the Gold-exclusion assertion, the guard's
selection on a real fact record, and the checkpoint round-trip. What is **not** verified is
anything about Llama-3.1-8B's actual behaviour — whether it reads a soft token, whether the
gate stays open, whether S1 beats A1. Those are the questions Phase 9 exists to answer and
none of them has been answered. Unblocking needs rented compute, teacher-API budget and an
annotator, in that order of lead time.

---

## D-069 — The fusion projector is fp32 and the check is an assertion, not a convention

**Date:** 2026-08-04 · **Phase:** 8/9 · **Status:** accepted
**Decision:** Every fusion parameter is held in `torch.float32`.
`models.fusion.base.assert_projector_is_fp32` runs after model construction and refuses any
non-fp32 floating-point parameter and any `bitsandbytes` submodule anywhere in the fusion
tree. It is called twice: inside `Graph2TextGenerator.__init__` and again in
`build_generator` after assembly.
**Rationale:** The projector is a randomly-initialised map learning to land inside an
already-trained embedding distribution. Quantising it to nf4 discretises exactly the
parameters that must move precisely for that to converge. The failure mode is not a crash —
it is a run that trains for fourteen hours and produces soft tokens the language model
reads as noise, with a loss curve that looks entirely healthy because the serialised facts
in the prompt carry the task on their own.

Twice, rather than once, because the window is real: `prepare_model_for_kbit_training` and
`get_peft_model` both walk the module tree casting and replacing layers, and
`modules_to_save` — the mechanism that makes the projector trainable at all — is precisely
what walks it. A projector that was fp32 at construction is not necessarily fp32 when the
first batch arrives.
**Consequences:** `tests/unit/test_fusion.py` asserts both refusals, including a fake
`bitsandbytes`-module check. A future change that casts the whole model with `.half()`
fails at construction rather than at the results table.

---

## D-070 — Three learning rates, because one produces a result indistinguishable from failure

**Date:** 2026-08-04 · **Phase:** 9 · **Status:** accepted
**Decision:** Trainable parameters are split into three optimiser groups with separate
rates: LoRA adapters 2e-4, fusion projector 1e-3, encoder 1e-5. The schedule is a
`LambdaLR`, which *scales* each group's own rate rather than setting an absolute one, so
the ratio holds for the whole run. Gradient norms are logged per group.
**Rationale:** The three components are at completely different points in their training.
The adapters modulate a converged 8B model. The projector starts from an `nn.Linear`
initialisation and has to travel to a specific region of a 4096-dimensional embedding
space. The encoder is already trained and was selected on val AUC-PR in Phase 7.

At a single 2e-4 the projector is still in transit when the adapters have converged. The
model therefore learns to solve the task from the serialised facts alone — and does so
*permanently*, because by the time the projector arrives the model has no remaining use for
it. The observable result is a healthy loss curve, a fluent narrative, and **S1 ≈ A1**:
the exact signature of "the architecture contributes nothing", produced by an optimiser
setting rather than by the architecture.

Per-group gradient norms are logged for the same reason. One aggregate norm cannot
distinguish "the projector is learning" from "the adapters are learning and the projector
is receiving nothing", and the second is a wiring bug that trains happily.
**Consequences:** `trainable_parameter_groups` is a method on the generator with a test
asserting all three rates arrive distinctly and that no parameter appears in two groups.
Should Gate 8 fail, the three-LR scheme is the first thing to re-examine before concluding
the architecture is at fault: a null result under a single LR would not be evidence.

---

## D-071 — The A1 control is a derangement, and a singleton batch draws from a ring buffer

**Date:** 2026-08-04 · **Phase:** 8 · **Status:** accepted
**Decision:** `ShuffledGraphFusion` pairs each narrative with a different case's graph by
sampling a **derangement** — a permutation with no fixed point — rather than a plain
`randperm`. Shuffling happens on the pooled tokens *before* projection. A batch of size 1
draws a foreign case from a ring buffer of the previous eight batches; when the buffer is
empty the batch is recorded as unshuffled in `ShuffleStats` rather than passed through
silently. `across_batch` is the configured mode; `within_case` and `noise` exist as weaker
and stronger nulls respectively.
**Rationale:** `torch.randperm` leaves fixed points. At the configured batch size of 2 it
returns the identity half the time, so a "control" built on it would hand roughly a quarter
of its cases their own graph. That is not a control — it is a partially-treated arm, and it
biases the S1-vs-A1 comparison **towards rejecting the null**, which is the direction that
produces a paper claiming a contribution it does not have.

Shuffling before projection rather than after keeps the projector's input distribution
identical across the two arms, so the comparison isolates the graph-to-narrative
correspondence rather than confounding it with a change to what the projector was trained
on.

The singleton case matters because the last batch of an epoch can be size 1, and there is
no derangement of one element. Leaving it unshuffled would quietly feed the control its own
graph on those steps.
**Consequences:** `ShuffleStats.n_fixed_points` **must be zero** and is asserted in the
tests over many batches at every batch size the project uses. It is reported in the run
record so the paper can state the property rather than assume it.

---

## D-072 — The within-run shuffle and the A1 arm are two different controls, reported separately

**Date:** 2026-08-04 · **Phase:** 9 · **Status:** accepted
**Decision:** `FaithfulnessCallback` measures a **within-run shuffle** — the model under
training, evaluated on the fixed probe cases with its graph tokens deranged at *inference*
time — and logs it as `shuffled_*` beside the model's own `probe_*` from step 0. The
**A1 arm** is a separately trained model and is compared by `compare_arms` /
`scripts/09b_compare_arms.py`. Gate 8 is decided on the second, never the first.
**Rationale:** The within-run shuffle is free, available every diagnostic step, and is the
intended early warning: if faithfulness is unchanged when the graph is scrambled, the run
should be stopped and diagnosed rather than left to burn fourteen hours. But it is not the
paper's control. A model trained on correct pairings can be robust to inference-time
scrambling for reasons unrelated to whether it needed the graph — it may have learned
graph-independent priors that the scramble does not disturb. Only a model *trained* on
deranged pairings answers the question Gate 8 asks.

Conflating them would be a reviewer's first catch, and the cost of keeping them apart is
one prefix in a log file.
**Consequences:** Two `tracking_*` parameters control when the callback raises
`tracking_alarm`, and it deliberately does **not** halt the run — an early run has both
curves near zero for ordinary reasons, and a callback that killed the job for that would be
worse than the problem. A human decides.

---

## D-073 — Guard weights 0.5 / 0.35 / 0.15, and coverage is in the score to stop the guard winning by saying nothing

**Date:** 2026-08-04 · **Phase:** 9 · **Status:** accepted
**Decision:** The inference guard scores candidates as
`0.5*(1 - contradiction_rate) + 0.35*coverage + 0.15*(1 - unverifiable_rate)`, with the
weights required to sum to 1.0. Coverage is measured against Phase 3's salience list for
the record's typology, with unsupported fields already excused (invariant 4). The guarded
and unguarded results are reported as **two separate rows** of the results table.
**Rationale:** The weights are a judgement call, not a measurement, which is why they are
recorded here. Contradiction is weighted highest because a contradicted claim is a false
statement in a regulatory filing — categorically worse than an incomplete one. Unverifiable
is weighted lowest because such a claim is usually a hedge or a stylistic phrase the checker
has no field for, not a falsehood.

Coverage is in the score because **contradiction rate alone is maximised by saying almost
nothing**. A one-sentence narrative naming the subject account and stopping has a
contradiction rate of exactly zero and would win every selection. Without a coverage term
the guard would systematically select the emptiest candidate and the guarded system's
faithfulness would rise while its usefulness collapsed.

The two rows are not a presentational choice. Raw model faithfulness is the *scientific*
claim — what the architecture achieved. Guarded faithfulness is the *application* claim —
what a deployment delivers, and it is higher partly because a verifier discarded the bad
candidates. Reporting the guarded number as the model's faithfulness credits the
architecture with the verifier's work.
**Consequences:** `GuardReport` carries `selected` and `unguarded` (the first sampled
candidate) from the same run, so the two rows cannot drift apart. `GuardStatistics` records
how often the guard changed the selection, triggered a regeneration, and failed and warned —
a guard that never intervenes costs 4× the compute for nothing, and that is a publishable
finding too.

---

## D-074 — vLLM is not used, and the reason is recorded rather than omitted

**Date:** 2026-08-04 · **Phase:** 9 · **Status:** accepted
**Decision:** Batch generation uses HuggingFace decoding over `inputs_embeds` with manual
batching, not vLLM. Throughput is measured and reported by `models.generator.profiling`.
**Rationale:** vLLM's fast paths key on token ids — the scheduler, the paged KV cache and
the prefix cache all do. This model's graph conditioning enters as *embeddings spliced at
reserved positions inside* the prompt, which vLLM has no supported entry point for:
`prompt_embeds` replaces a whole prompt rather than a span within one, and the soft tokens
differ per case so there is no shared prefix to cache. Supporting it means either
reconstructing the splice inside a custom vLLM worker or giving up the graph pathway.

Decoding is written as an explicit loop rather than through `model.generate` because the
splice must happen once, before the first forward pass, and not be recomputed on
continuation steps. Expressing that through `generate`'s hooks is more fragile than the
loop, and the loop runs against a stub backbone, which is what makes the guard's selection
logic testable without a GPU.
**Consequences:** Generation is slower than a text-only baseline served on vLLM, and B7 —
which has no soft tokens — *could* use vLLM. It deliberately does not: an arm served by a
different engine is not throughput-comparable, and Phase 13 reports throughput. The cost is
recorded as a limitation rather than hidden.

## D-075 — Tolerance policy is confirmed unchanged, and evaluation reuses the checker rather than restating it

**Date:** 2026-08-05 · **Phase:** 10 · **Status:** accepted
**Decision:** Phase 10 introduces **no new tolerance, threshold or comparison rule.** Every
verdict in Layer 2 comes from `g2t_aml.facts.checkers.check_claim` under the Phase 3
`ToleranceConfig` exactly as frozen: counts exact, money 1% relative with an absolute
floor, durations one unit of the granularity the narrative itself stated (D-027),
timestamps one minute, shares within `share_absolute`. Salient fields come from
`facts.salience.salience_report` and are not redefined. The hallucination classes come
from `facts.taxonomy` and are not extended.
**Rationale:** The corpus verifier and the evaluation metric are deliberately the same code
run in opposite directions (CLAUDE.md §3). A second tolerance policy on the evaluation side
would mean Bronze could be verified as faithful at build time and scored as unfaithful at
evaluation time, or the reverse — and either way the disagreement would be a *parameter*
rather than a bug, which is precisely the property the single-implementation design exists
to prevent. It also means a tolerance can never be relaxed to improve a result: relaxing it
regenerates the corpus.

The one thing Phase 10 does add is *attribution* —
`eval.claim_extraction.deterministic` binds a quantity to a field so the existing checker
can adjudicate it. That changes which claims reach the checker, never what the checker does
with one. See D-076 for why that asymmetry is safe.
**Consequences:** No `eval/` module may hold a numeric threshold that decides a verdict. The
two thresholds `eval/` does own — the template-baseline margin (0.02 ROUGE) and the
extractor-agreement match IoU (0.5) — decide a *finding* and a *pairing*, not a verdict,
and both are fixed in code before any system was scored. `tests/unit/test_eval_layer2.py`
pins the arithmetic; `tests/integration/test_eval_end_to_end.py` pins that Bronze scores
1.000 under it.

## D-076 — Cue attribution can only sharpen a verdict, never soften one

**Date:** 2026-08-05 · **Phase:** 10 · **Status:** accepted
**Decision:** Method A adds a table of surface cues (`DEFAULT_RULES`) that bind an
unaligned quantity to a fact field. A cue that fires produces a claim naming that field,
which the checker resolves to SUPPORTED or CONTRADICTED. A cue that does not fire leaves
the claim naming no field, which stays UNVERIFIABLE. **The pass has no path that turns a
CONTRADICTED claim into anything softer**, so a missing rule costs sensitivity and never
correctness.
**Rationale:** Phase 5's extractor was built for a repair loop where UNVERIFIABLE and
CONTRADICTED have the same consequence — both exhaust the budget and the record is repaired
or discarded. Phase 10 reports them as different numbers, and Hallucination Rate is
*contradicted over total*. Carried over unchanged, a system writing "the subject received
from 14 distinct counterparties" against a record saying 9 would score **zero
hallucinations and one unverifiable claim** — the most damaging thing a SAR narrative can
do, filed under the bucket for things the graph merely cannot speak to. Measured on the
fan-out fixture: without attribution that narrative reports Hallucination Rate 0.000 and
Zero-Hallucination 1.000; with it, 1.000 and 0.000.

The deliberate non-decision is the mirror image: an unaligned quantity that matches **no**
cue is *not* promoted to H2. H2 is a number that disagrees with the record; a number the
record cannot speak to has not disagreed with anything. Promoting it would move the largest
single bucket of unverifiable claims into the hallucination count and roughly double every
reported hallucination rate for no reason but a classification choice.

**A second, related hole was found by the H6 test and closed the same way.** Phase 5 matches
only *whitelisted* citations, deliberately, so it can never launder an invented one — but
that leaves the complement uncovered, and "the USD 42,000 mandatory disclosure threshold"
produced one stray number and no regulatory claim at all. The Critical class the paper
leans on hardest was unreachable. `_REGULATION_RE` now matches the **shape** of a citation
regardless of content and lets `check_regulatory` adjudicate against the whitelist: a list
of forbidden citations cannot contain the one a model has not invented yet.
**Consequences:** The cue table is a sensitivity surface and must be reported as one. A
narrative phrased outside every cue has its quantities scored UNVERIFIABLE rather than
checked, which *understates* hallucination — the conservative direction, and the one that
shows up in the unverifiable rate where a reader can see it. `DeterministicReport.attributed`
records which rule fired on every claim so a systematic mis-attribution is traceable to one
rule rather than to "the extractor".

## D-077 — Zero-Hallucination Rate is the headline, and averaged faithfulness is diagnostic

**Date:** 2026-08-05 · **Phase:** 10 · **Status:** accepted
**Decision:** The per-narrative binary — *does this narrative contain no contradicted claim
at all* — is the number the paper leads with, the first row of every table, the first key
in the JSON, and the metric the pairwise significance tests are run on first. Fact
Precision, Hallucination Rate and Fact F1 are reported below it as diagnostics.
`report.HEADLINE_METRIC` names it once so the three emitters cannot disagree about what
"the headline" is.
**Rationale:** A SAR narrative containing one fabricated fact is unusable regardless of how
good the rest of it is. It cannot be filed, and an investigator who finds one error has to
re-verify the whole thing — the cost is not proportional to the error rate, it is a step
function at the first error. Averaged precision cannot express that. Two systems at 90%
mean Fact Precision are, at the extremes, a system that puts exactly one error into every
narrative (Zero-Hallucination 0%) and a system perfect on nine narratives in ten and badly
wrong on the tenth (90%). Those are different products — one is undeployable and one has a
triage problem — and the mean reports them as identical.
`tests/unit/test_eval_layer2.py::test_zero_hallucination_is_per_narrative_not_averaged_precision`
constructs exactly that pair and asserts the two metrics separate them.

The same reasoning makes **Critical Error Rate per-narrative** rather than per-claim: one
fabricated regulation makes a report unfileable, so "2% of narratives cite a rule that does
not exist" is the sentence a compliance reader acts on and "H6 is 0.3% of claims" is not.

For an applications venue this framing is also the more persuasive one: it states the metric
in the unit the practitioner works in — the document — rather than in the unit the model
works in.
**Consequences:** Aggregation is a **macro mean over narratives**, not a pooled ratio over
claims, so one long narrative with two hundred claims cannot dominate a hundred short ones.
The pooled figures are computed and reported beside the macro ones
(`pooled_fact_precision`, `pooled_hallucination_rate`) so the difference is visible rather
than taken on trust. A narrative from which no claim could be extracted scores perfect
precision *and* perfect zero-hallucination, so `n_narratives_with_no_claims` is reported
prominently: a rising count there is the signature of an extractor failure masquerading as
a quality improvement.

## D-078 — Two claim extractors, and the agreement between them is a reported number

**Date:** 2026-08-05 · **Phase:** 10 · **Status:** accepted
**Decision:** Layer 2 is computed by a deterministic extractor (Method A) and validated
against an independent LLM-based one (Method B) on a 300-case sample. **The two share no
machinery** — Method B does not call `check_claim`, does not see the Bronze slot table and
does not consult the controlled vocabulary; it reaches its own verdict from the serialised
fact record used as NLI premises. The only thing they share is the three-valued verdict
vocabulary. Three agreements are reported: verdict κ over matched claims, boundary κ over
token-level claim membership, and decision κ on the per-narrative zero-hallucination binary.
**Rationale:** Extraction is where a faithfulness metric is most easily wrong and least
easily seen to be wrong. An extractor that quietly fails to find a claim reports a narrative
as more faithful than it is, and nothing about the resulting number looks unusual. A single
extractor validated against nothing produces a metric with a method section but no evidence.

Two extractors that both consulted the checker would agree *by construction* and the κ would
measure nothing, which is why Method B is kept away from it entirely. Method B also runs as
**two calls, not one**: a single call that both decomposed and adjudicated would let the
model settle on a verdict first and then choose a decomposition supporting it, biasing
exactly the claim boundaries the boundary κ measures.

Boundary agreement is token-level rather than span-level because a span-level κ needs a
chance model over an unbounded set of possible spans, which does not exist. Decision
agreement is reported separately because it is agreement on the number actually published,
and it can differ sharply from claim-level κ — a hundred claim-level disagreements spread
across a hundred narratives that each still contain some other contradiction change the
headline not at all.
**Consequences:** Method B needs teacher-API credentials and is **off by default**
(`eval.agreement.enabled: false`); the harness and its whole test suite run in CI without a
network or a key, through `ScriptedTeacher`. **The κ has therefore not been measured** — see
the Phase 10 log entry, which lists it as the phase's one unmet acceptance criterion and
names what unblocks it. The machinery is complete and tested end to end on scripted
responses; what is missing is the spend authorisation, the same blocker as Phase 5.

## D-079 — Wilcoxon, Holm, Cliff's δ, and the correction family is the function's own output

**Date:** 2026-08-05 · **Phase:** 10 · **Status:** accepted
**Decision:** Paired significance is **Wilcoxon signed-rank**; multiplicity is corrected by
**Holm-Bonferroni**; the primary effect size is **Cliff's δ**, with Cohen's d reported
beside it. Confidence intervals are **percentile bootstrap at 10,000 resamples**. The
correction family is **defined as one call to `compare_systems`** — every pairwise
comparison of one metric on one test stream — and the family's size is written onto every
comparison it returns.
**Rationale, test by test.**
*Wilcoxon over a t-test:* the per-case metrics are bounded rates, several sit against their
ceiling, and Zero-Hallucination is a per-narrative binary. None is remotely normal.
*Holm over plain Bonferroni:* uniformly more powerful at the same familywise error rate.
*Holm over Benjamini-Hochberg:* the claims here are asserted individually — "S1 beats B7 on
Zero-Hallucination" stands on its own, not as one of a set whose false-discovery
*proportion* is controlled.
*Cliff's δ over Cohen's d:* δ assumes nothing about the distributions, which is what a
bounded rate piled against a ceiling requires; d is computed anyway because reviewers ask
for it, and showing both makes a disagreement between them visible.
*Percentile over BCa:* several metrics sit on a boundary, where BCa's acceleration term is
estimated from a jackknife that is itself degenerate.
*10,000 over the 1,000 the config previously carried:* at a thousand resamples the interval
endpoints move in the third decimal between seeds, and the third decimal is the precision
the tables quote — a re-run would move a published bound for no reason but the resampling.

**Defining the family as the function's own output is the load-bearing part.** With sixteen
systems there are 120 pairwise comparisons per metric; at α = 0.05 six are expected to be
"significant" under a complete null. A family assembled by a caller across several calls is
a family nobody can reconstruct from the results file, and the difference between correcting
over 120 comparisons and over 15 is the difference between a finding and a coincidence.
**Consequences:** Three rules are enforced by the type shapes rather than by remembering
them. `seed_summary` returns `std=None` for a single seed, so a single-seed cell prints
"(1 seed)" rather than "± 0.0000". `PairedComparison` has no constructor path producing a
p-value without an effect size. `PairedComparison.significant` is **False when
`p_adjusted` is None**, so a comparison that bypassed the family can never be reported as
significant — tested in
`test_an_uncorrected_comparison_is_never_reported_as_significant`. `holm_bonferroni` is
tested against `statsmodels.multipletests(method="holm")` on a hand-written family and on
random families of size 2, 5, 17 and 120.

## D-080 — Layer 1 leads nothing, and the Bronze template's scores are an output rather than a footnote

**Date:** 2026-08-05 · **Phase:** 10 · **Status:** accepted
**Decision:** Surface-overlap metrics are computed against **Gold references only**, are
reported **after** Layer 2 in every artifact the harness emits, get **no bootstrap interval
and no significance test**, and are accompanied by `template_baseline_finding` — a
mechanical comparison of the deterministic Bronze template against the best model arm,
which flags the metric as **non-discriminative** when the margin falls below 0.02 ROUGE-L.
**Rationale:** The order a results section is read in is the order its numbers are believed
in; an evaluation whose first table is BLEU has told the reader which number matters before
it has argued for one. Giving overlap metrics the same statistical dress as the faithfulness
metrics — intervals, corrected p-values — invites the reader to weigh them equally, which
is the opposite of this project's argument.

The template comparison is the reason the harness computes Bronze's Layer 1 scores at all.
Bronze has no model in it. If it scores competitively against Gold on ROUGE, then ROUGE does
not distinguish a system that understands a case from one that fills in blanks, and every
overlap number in the agentic-SAR literature is measuring something other than what it
claims to. That is a quotable result, so it is produced **by the harness and flagged**,
rather than left as a note someone remembers to write. The 0.02 margin is fixed in code
**before any system was scored**; a threshold chosen after seeing the results is not a
threshold.
**Consequences:** Every Layer 1 metric is individually optional and records *why* it is
absent in `Layer1Metrics.unavailable`, never a zero — BERTScore needs a 1.4 GB model,
METEOR a WordNet download, and a BLEURT/COMET-class metric weights that are not in this
repository's dependency set at all (both are supplied through a protocol, not a dependency).
A harness that raised on a missing model would be unrunnable in CI; one that returned zero
would put a zero in a results table. BERTScore is **always** rescaled with baseline and that
is not exposed as a parameter, because an unrescaled BERTScore is uninterpretable across
papers and a parameter that let one not be would eventually be set. The BLEU signature is
returned *with* the score rather than logged, and self-BLEU carries its reference count
(D-043).

**Bronze's overlap scores have not been computed**, because Gold does not exist. That is
recorded in the Phase 10 log entry as a deferral with the finding it blocks, and the harness
prints `no case has a Gold reference` rather than a number.

---

## D-081 — The seed asymmetry: three seeds on four systems, one everywhere else

**Date:** 2026-08-05 · **Phase:** 11 · **Status:** accepted
**Decision:** The experiment matrix runs **three seeds (42, 1337, 2024) on S1, S2, A1 and
B7** — the systems carrying the central claim — and **one seed (42) on the other thirteen
systems**. The asymmetry is stated in the paper, marked in every table with a dagger and in
every figure with a hollow marker, and enforced by `registry.validate_registry`, which
refuses a central-claim system at fewer than three seeds and a non-central system at more
than one.

**Rationale:** The full seventeen-system matrix at three seeds is roughly 3–5 GPU-weeks and
fits no schedule this project has. The alternative to an asymmetry is not symmetry — it is
a *smaller matrix*, dropping ablations to afford variance estimates on arms nobody
disputes. Between "every system at one seed" and "the systems carrying the claim at three,
the rest at one", the second puts the variance estimate exactly where a reviewer will ask
for it: on S1 vs A1 (Gate 8), on S1 vs B7 (does the graph beat serialised facts), and on
S2 (the headline).

The failure mode this creates is a single-seed number read as if it had a variance
estimate. That is closed mechanically rather than by convention: `SeedSummary.std` is
`None` at one seed by construction, `_mean_std_cell` renders `None` as a bare mean with a
dagger and never as `± 0.0000`, and `figures._series` returns a per-system single-seed flag
that the bar chart draws. A single-seed row cannot print a zero standard deviation.

**Extension order if compute frees up: A2, then B8**, recorded in
`registry.SEED_EXTENSION_ORDER`. A2 answers "is topology needed" against a control that
already reaches 0.80 AUC-PR without message passing, and B8 is the published-baseline arm;
both currently carry a claim on one seed. Extending is `registry.with_seeds`, a call with a
recorded argument, rather than an edit to the table.

**Consequences:** Thirteen of seventeen systems have no variance estimate, and no claim in
the paper may rest on a difference between two single-seed arms. Where such a comparison is
interesting — A3_F3 against A3_F4, say — it is reported as a difference without a
significance claim, and said to be that.

---

## D-082 — F1–F4 are named against the built machinery, and A3's F1 point is B8

**Date:** 2026-08-05 · **Phase:** 11 · **Status:** accepted
**Decision:** The brief's fusion variants map onto what Phase 8 actually built — one gate
flag and three projector kinds — as: **F0** no fusion; **F1** ungated MLP; **F2** gated MLP;
**F3** gated linear; **F4** gated perceiver resampler. The mapping is declared on
`registry.FusionVariant` with `gated` and `projector` properties, and
`tests/integration/test_matrix_pipeline.py` asserts that every arm's composed Hydra config
agrees with what the registry claims.

**The A3 ablation's F1 point is B8**, not a separate arm.

**Rationale:** Phase 8 implemented `PrefixFusion(gated: bool)` and three projectors, not
four numbered variants. Leaving F3 and F4 undefined would have meant each reader inventing
their own reading of the ablation table; defining them in a docstring only would have left
the registry and the configs free to disagree. Naming them where the matrix is declared,
and testing the configs against the names, makes "which arm is F4" answerable from one
place.

F1 and F2 differ in **exactly one flag**, which is what makes the gate's contribution
measurable rather than confounded with two separately-tuned models. F3 and F4 hold the gate
at F2's setting and move only the projector, so the two axes are never varied together.

B8 is already `gated=false, projector=mlp, text_mode=full`, the same base model and the same
training regime — which is precisely what an A3 F1 arm would have been. Running it twice
under two names would spend GPU-days to produce a duplicate row. The ablation table
therefore reports {B8 (F1), S1 (F2), A3_F3, A3_F4} and says so in its caption.

**Consequences:** The fusion ablation has four points, not three, and one of them is shared
with the baseline table. A reader comparing the two tables sees B8's number twice; the
caption states why.

---

## D-083 — Baseline model versions and dates are registry data, and a stale one fails validation

**Date:** 2026-08-05 · **Phase:** 11 · **Status:** accepted
**Decision:** Every system's base model, its exact pinned version and its **release date**
are fields on `SystemSpec`. `validate_registry` refuses any baseline whose release date is
before 2024-01-01, and refuses any named model with no date at all. The frontier arms
(B3–B5) use `claude-opus-5` (released 2026-02-05); the local arms use
`meta-llama/Llama-3.1-8B-Instruct` (2024-07-23); A6's second base is `Qwen/Qwen3-8B`
(2025-04-29). A test additionally requires B3–B5 to be 2025 or later.

**Rationale:** Stale comparison tables are a documented desk-reject trigger at this venue,
and a table whose baseline says "GPT-4" without saying which one is a table a reviewer
cannot reproduce. Recording the version in prose means recording it in five places that will
eventually disagree. Recording it as validated registry data means the check runs before any
compute is spent, and the paper's methods section is generated from the same field.

The floor is a *validation* rather than a note because staleness arrives by inaction: a
matrix configured today and run in eight months has an eight-month-old baseline and nothing
in the run will mention it.

**Consequences:** Changing a baseline model is a registry edit that a test will notice, and
the date must move with the identifier. When these models age out, `validate_registry`
does not fail — the floor is 2024 — so the 2025 floor for the frontier arms is asserted in
`test_frontier_baselines_are_2025_or_later`, which is where the currency requirement
actually lives.

---

## D-084 — B5 verifies itself, and is given more inference compute than any of our arms

**Date:** 2026-08-05 · **Phase:** 11 · **Status:** accepted
**Decision:** B5, the agentic comparator, runs generate → **self-verify** → repair for up
to **three** rounds, starting from the *few-shot* draft rather than the zero-shot one. Its
verification is performed by its own model against the serialised record, through
`prompts/baseline_verify_v1.txt`. **Our Phase 3 checker never runs inside the loop.** The
call count per narrative is recorded on `AgenticTrace.n_calls` and reported.

**Rationale:** Three separate reasons, and they point the same way.

*It has to be the method that exists.* B5 stands in for the published agentic-SAR approach,
in which the system's own LLM verifies. Handing it our three-valued checker would build a
competitor nobody can cite and would answer a question nobody asked.

*It would poison the comparison in the other direction.* Our checker is the instrument the
evaluation scores every system with. A baseline that optimises against the scorer during
generation, when our own arms do not, is not a stronger baseline — it is a different
experiment.

*A competitor is entitled to its best configuration.* Hence the few-shot starting draft,
three repair rounds rather than Silver's two, and the regulatory whitelist supplied to the
verifier — a verifier not told the whitelist cannot catch an invented citation, and H6 is
the Critical class the paper leans on hardest.

The consequence is that B5 spends several model calls per narrative where S1 spends one
forward pass. **That asymmetry favours B5 and is reported rather than corrected away.**

Two smaller decisions inside the loop, both in the same direction:
- A verification response that does not parse is recorded as a **parse failure** and ends
  the loop. Retrying until the format is right is a loop that selects for whichever answer
  happens to parse, and it would inflate B5's self-reported convergence.
- The narrative returned is the **last** one produced, converged or not — never the
  best-scoring draft under our checker, which would be selecting the baseline's output with
  our instrument.

**Consequences:** B5 is the most expensive arm in the matrix and its cost is capped at
$600. Its self-reported clean rate and its measured Zero-Hallucination Rate are separate
columns and are never conflated: the first is a property of B5's self-assessment, the
second comes from the instrument every system is scored with.

---

## D-085 — Resumption is a completion marker AND a config hash, and the hash is in the path

**Date:** 2026-08-05 · **Phase:** 11 · **Status:** accepted
**Decision:** A matrix run counts as complete when its directory holds `COMPLETED.json`
**and** that file's recorded `config_hash` matches the config the run would be given now.
The run directory is `<root>/<system>/seed<n>/<config-hash-prefix>`. Fields that describe a
row rather than determine a computation — `role`, `description`, `notes` — are excluded
from the hash input.

**Rationale:** Each half alone fails in a different direction. A marker alone would skip a
run whose config has since changed, **reporting an old number under a new configuration** —
which is precisely how a results table becomes irreproducible without any single number in
it being wrong. A hash alone would re-run everything that had completed, which on a
GPU-week matrix is not a resumption strategy.

Putting the hash in the *path* makes invariant 6 mechanical rather than remembered: a
re-run under a changed config cannot land on top of the old one, because it resolves to a
different directory. The old number stays on disk and stays readable.

Excluding the descriptive fields is what stops a prose edit costing a GPU-week. Editing a
`role` string is editing a table caption, not an experiment.

**Consequences:** Several config hashes can coexist under one seed; the aggregator reads
the most recently completed and leaves the others in place. A truncated marker — what a
killed job leaves — parses as incomplete rather than as complete, and the run is redone.

---

## D-086 — Failure isolation, and a partly-failed dependency is not a dependency

**Date:** 2026-08-05 · **Phase:** 11 · **Status:** accepted
**Decision:** A failing run is recorded with its full traceback in its own directory and in
the matrix summary, and the matrix continues. A system whose *dependency* failed is marked
**blocked** and not attempted. A system counts as a satisfied dependency only when **every
one of its seeds** succeeded or was skipped.

**Rationale:** The matrix is twenty-five runs, several of them GPU-days apart, on hardware
that will be interrupted. Aborting on the first failure means one bad arm costs the other
twenty-four.

The dependency rule is the part that is not obvious. A5 reads S1's checkpoint. If S1
succeeded at seed 42 and diverged at 1337, an A5 that runs anyway reads whichever seed
happened to survive — and produces a number, under a row heading that says it is S1's
checkpoint with the guard off. **Running it is worse than not running it, because not
running it is visible.** So a partly-failed system blocks its dependents, and the blocker
is named in the record.

**Consequences:** `MatrixResult.ok` is false whenever anything failed or was blocked, and
`scripts/11_run_matrix.py` exits 2 on that — a matrix that lost three arms overnight must
not look to CI like a matrix that finished. The aggregator reports every missing run by
name and reason rather than quietly producing a shorter table.

---

## D-087 — Phase 11 was built but not run, and RESULTS.md ships with the absences

**Date:** 2026-08-05 · **Phase:** 11 · **Status:** accepted
**Decision:** The Phase 11 orchestration, aggregation, statistics, figures and qualitative
tooling are complete and tested. **No system in the matrix has been run.** `RESULTS.md` is
committed now, carrying every measured number the project has and every declared run as a
named absence with its blocker.

**Rationale:** The Phase 11 preflight failed. Phase 9 never ran — D-068 — so there are no
trained arms to compare, and there is no S1-vs-A1 outcome to confirm or reframe from. The
blockers are unchanged: a 4 GB card against a model needing ≥24 GB, zero Silver for want of
API credentials, zero Gold for want of an annotator.

Everything that does not depend on those was built, because the value of this phase is that
the matrix runs on the *first* day compute lands rather than starting a week of integration
then. The aggregation and figure paths are deliberately exercised against an empty matrix
and against fixture metrics, which is the only way to know they work before the numbers
exist.

Committing `RESULTS.md` with nothing but absences in its main table is invariant 7 applied
to a phase that produced no results: the file's job is to answer "did you try X", and
"declared, configured, blocked on a GPU" is an answer. It is also a commitment device — the
sixteen rows are in the table now, so a system that later produces an inconvenient number
cannot quietly fail to appear.

**Consequences:** Every acceptance criterion of the Phase 11 gate that depends on a run is
**deferred, not met**, and is listed as such in `PHASE_LOG.md`. No number in this phase's
log is a result about the method. What is untested is stated rather than papered over: the
GPU and API executors have never been executed against a real model or a real endpoint, and
their first run will find integration bugs the CPU path did not.

---

## D-088 — Every rater rates a shared anchor block, or there is no agreement statistic

**Date:** 2026-08-05 · **Phase:** 12 · **Status:** accepted
**Decision:** The Phase 12 block design reserves a small set of (case, system) cells that
**every** rater rates, in addition to their own case-disjoint block. Fifteen cells at a
panel of ten, spread evenly across systems, rounded to a whole multiple of the system count.

**Rationale:** The rest of the design is deliberately optimised for breadth. No rater sees
a case twice, and the allocator picks the least-covered cell at every step, which maximises
the number of distinct (case, system) cells the panel touches and therefore the number of
cases each arm is judged on. That is the right objective for the between-system comparison
and it has one fatal consequence: **almost no cell is rated by two people.**

Krippendorff's α is defined over units that two or more coders judged. A design optimised
purely for breadth yields *zero* pairable units, and the α the Phase 12 gate requires
cannot be computed at all — not poorly, not with a wide interval, but not at all.

This was not found by inspecting the design. It was found by running the analysis against a
simulated response set and reading the warning the analysis emitted about itself: "no
(case, system) cell was rated by two raters, so inter-rater agreement is not estimable from
this design as run". The design's own report looked healthy — balanced workloads, good
position balance, 85 cases per system.

The anchor block is rounded up to a multiple of the number of systems. Twelve anchors
across five systems gives 3/3/2/2/2, and that unevenness becomes a permanent per-system
offset in every rater's workload, which the balanced-workload validator then rejects.
Rounding costs at most k−1 items of everyone's time and makes the arithmetic exact.

**Consequences:** Fifteen items of every rater's workload contribute nothing to the
between-system breadth. At 60 items that is a quarter of the anchor-adjusted budget, and it
is why the recommended panel is 10 raters rather than 8 — the gate's 80-cases-per-system
floor is only reached at 10 × 60 once the anchors are paid for.

The trade is worth it because the alternative is not "a slightly weaker agreement
statistic", it is a results table of Likert means with no agreement statistic beside them,
which the brief forbids and which a reviewer should refuse.

`test_the_anchor_block_makes_agreement_computable` is the regression test, and it asserts
on the *analysis output* rather than on the design, so it fails if either end breaks.

---

## D-089 — Friedman is reported on rater blocks, and Durbin on case blocks, and both are published

**Date:** 2026-08-05 · **Phase:** 12 · **Status:** accepted
**Decision:** Two omnibus tests are computed and reported for every metric, not one.
`friedman_test` runs over **rater-blocked means**; `durbin_test` runs over the
**case-blocked incomplete design**. Where they disagree, the disagreement is reported as
the finding.

**Rationale:** The brief specifies a Friedman test followed by Nemenyi. Friedman requires
**complete blocks** — every block carrying one observation per treatment — and this design
cannot supply them. No rater sees the same case twice (which is the anchoring protection
the design exists for), so no case carries one observation per system, and the natural
blocking variable is unavailable.

Two ways out, and each is deficient alone:

- **Block by rater.** Each rater contributes one mean per system, giving a genuinely
  complete blocks-by-treatments matrix. This is the brief's test and it is defensible. But
  it has as many blocks as there are raters — eight or ten — so it is badly underpowered,
  and it discards the item-level variation by averaging it away.
- **Block by case and use Durbin's test**, which is Friedman's generalisation to a balanced
  incomplete block design and is the textbook answer to exactly this situation. It uses
  every item-level observation. But it is not what the brief asked for, and its null
  distribution assumes balance in both directions, which a greedy allocation does not
  guarantee.

Reporting only the first throws away most of the data. Reporting only the second answers a
question the brief did not ask and would look like a substitution made to obtain a better
p-value. So both are reported, with the blocking variable and the block count printed
beside each statistic.

Durbin's implementation is validated by asserting it **reduces to Friedman on a complete
matrix**, exactly, rather than by trusting the algebra.

Because the case blocks are not automatically balanced, `_balanced_subset` takes the modal
block size and drops the least-represented systems until every remaining system appears in
an equal number of blocks. **The subset is named on the result and in the study's warnings**
— on the simulated data it was 77 of 100 case-blocks. A subset analysis reported as though
it were the whole is the failure mode here, not the subset itself.

**Consequences:** The results table carries two omnibus rows per metric and two
critical-difference diagrams. That is more apparatus than a paper usually shows, and the
alternative is a single test that is either underpowered or inapplicable, with the reader
unable to tell which.

If the internal-review fallback is used, the panel drops to 2–4 people and the rater-blocked
Friedman becomes meaningless. The case-blocked Durbin then carries the entire inferential
weight, and `docs/human_study/fallback_governance.md` says so explicitly.

---

## D-090 — Time-to-usable-draft is measured by two clocks, and which one produced it is stored

**Date:** 2026-08-05 · **Phase:** 12 · **Status:** accepted
**Decision:** Active time per item is measured by a browser-side component that can see
`visibilitychange`, with a server-side `BlurAwareTimer` as a fallback. Every response
records `timing_source` as `"browser"` or `"server"`, and the analysis reports the two
separately rather than pooling them.

**Rationale:** Time-to-usable-draft is one of the two measurements the study exists to
produce, and the strongest evidence available for a deployment claim. It is also trivially
corruptible: a rater who switches tab for four minutes is invisible to Python. Streamlit
does not re-run when a tab is hidden, no widget changes, and nothing reaches the server, so
a naive server-side clock counts the interruption as reading time. A handful of those move
a mean further than the effect being measured.

`visibilitychange` fires in the browser and nowhere else, so the measurement has to happen
there. The component is one static HTML file talking Streamlit's postMessage protocol
directly — no build step, no node, no bundled JavaScript in the lockfile — because it is a
released artifact and readable-in-a-text-editor is worth more than framework ergonomics.

**The fallback must not be silent.** If the component fails to register, the study should
still collect data; a session that dies because a timer did not load is worse than a session
with approximate times. But a set of times collected without visibility tracking *means
something different*, and pooling the two would put coffee breaks into the headline number
with nothing to show it happened. Hence the per-response flag, the count in the release
manifest, the warning in the analysis output, and the paragraph in the release README
telling a re-analyst they may drop those rows.

The server-side timer is the one that is unit-tested, because it is driven by explicit
timestamps and its arithmetic can be checked without a browser: eleven tests cover single
and multiple hidden periods, reading a running clock, stopping while hidden, double-blur,
and idempotent start under Streamlit's rerun-on-every-interaction model.

**Consequences:** Neither clock has yet run in a real browser, because there are no stimuli
to render. The first real session will find integration bugs the CPU path did not, and if
the component turns out not to register at all, the study still yields data — flagged, with
its times documented as upper bounds.

---

## D-091 — Free text is dropped from the release, and corrections are scanned and withheld

**Date:** 2026-08-05 · **Phase:** 12 · **Status:** accepted
**Decision:** The anonymised release deletes free-text comments entirely, publishes the
corrected narratives after passing each through the Phase 4 identifier scanner, withholds
any that trip it, and names the withheld items in the public manifest. Rater identifiers
are re-derived under a salt **not shared with the design**, and the mapping is written
outside the release directory.

**Rationale:** Four distinct re-identification routes, and each needs a different answer.

**Comments cannot be scrubbed.** "I've seen this pattern at my last employer" is identifying
and no pattern-matching rule catches the general case. A filter that removes what it
recognises leaves what it does not and produces a false assurance, so the field does not
survive at all.

**Corrections must be published anyway.** Edit distance is a headline measurement, and a
release from which it cannot be recomputed is not reproducible. They are therefore kept, but
scanned first with the deliberately blunt Phase 4 scanner — the one that would rather refuse
a clean narrative than pass an identifier, which is the correct bias in both places.
Anything flagged is withheld, **and its item id and reason go into the manifest**, so the
released count reconciles against the design and the exclusion is visible rather than
silent. A release that drops rows quietly is a release whose n cannot be checked.

**`rater-03` is already a pseudonym and that is not enough.** It is the *same* pseudonym
that appears in the recruitment records, the payment schedule and the consent forms, so
anyone holding those can join them to the release. The published labels are re-derived
through a keyed digest and sorted by digest, so `R01` is not `rater-01` and the ordering
carries nothing. Sharing the design salt would let anyone holding the design invert the
mapping, so `prepare_release` refuses when the two salts match.

**The mapping is written to the parent directory, marked PRIVATE.** Putting it inside the
bundle would undo the entire re-pseudonymisation the moment the bundle was zipped and
uploaded — which is a realistic accident, not a hypothetical one. The data management plan
requires it to be destroyed on publication.

**Consequences:** The published dataset cannot answer questions about rater commentary, and
a re-analyst cannot recover which internal rater is which. Both are intended. Participants
who decline the publication consent are excluded from the deposit and retained in the
analysis, which is why that consent is a separate initialled box rather than part of the
participation consent.

---

## D-092 — Phase 12 was built and not run, and the ethics application is the critical path

**Date:** 2026-08-05 · **Phase:** 12 · **Status:** accepted
**Decision:** The Phase 12 design, interface, analysis and release machinery are complete
and tested. **No human has rated anything.** The phase is recorded as built-and-blocked,
with a decision trigger date of **2026-09-15**.

**Rationale:** The preflight failed on three independent counts, and they are not the same
kind of problem.

**Ethics approval had not been applied for.** A search for "irb", "ethics" or "ethical"
across every markdown file in the repository returned zero matches. Approval takes 4–8
weeks and blocks all data collection. This is the binding constraint and, unlike the other
two, it is entirely within the project's control to start.

**One of five arms has generations.** Only Bronze exists. S1 and S2 need a GPU the project
does not have (D-068); B7, B3 and Silver need API credentials, the same blocker as Phase 5.
`scripts/12_build_study.py` exits non-zero rather than building a design over one arm,
because a design produced for a single system would spend a panel's time comparing that
system against itself before anyone noticed.

**No raters are recruited.** Phase 6 recorded recruitment as its critical path and it has
not moved.

Everything not depending on those was built, for the same reason Phase 11 was: the value is
that the study runs on the *first* day the inputs land rather than starting a fortnight of
integration then. Building it also produced three real defects (see the Phase 12 log) that
would otherwise have been discovered with a live panel — including one, D-088, that would
have made the gate's required agreement statistic uncomputable after all the data was in.

**The trigger date is a commitment device.** It is set on *submission*, not approval,
because an application submitted on time but approved late leaves the project waiting with
everything ready, which is recoverable, whereas an application never submitted is not. It
is recorded now so the choice is made on a calendar rather than in the week the deadline
arrives, when the temptation is to run something ungoverned and describe it generously.

**Consequences:** Layer 3 — the venue differentiator, and the thing whose absence is a
common cause of major-revision outcomes at *Expert Systems with Applications* — currently
has no results. If the fallback is used, the paper reports an internal expert review under
that name, with its four limitations in its own subsection, and never as a "user study" or
an "expert evaluation". If neither route runs, Layer 3 is reported as not performed with
its reason, which is invariant 7 applied to a phase that produced no results and is what
the project has already done for Phases 9 and 11.

---

## D-093 — Cost is a declared amortisation model, and the declaration is the deliverable

**Context.** Phase 13 has to put a "cost per 1,000 narratives" column next to a local
system and an API baseline. Those are not the same kind of number and the table will be
read as though they are.

An API price is **marginal**: one more narrative costs one more narrative's tokens, and
zero narratives cost nothing. A local cost is **capital already spent**, and at low volume
it is dominated entirely by the box existing. Printing 0.0005 beside 22.65 without saying
so is the standard way this comparison misleads, and it misleads in our favour, which is
exactly when a claim needs the most scaffolding.

**Decision.** The cost model is a dataclass whose every input is a field with a source,
not a number computed behind a function. `CostAssumptions` carries the capital figure, the
depreciation life, the utilisation factor, the power draw, the PUE, the electricity rate
and the API per-million prices; `hourly_cost()` shows its working; and the resolved
`amortised_usd_per_hour`, `power_usd_per_hour` and `hourly_cost_usd` are serialised into
every stored table so a reader never has to recompute a figure to check it.

`CostEstimate.basis` is `amortised_local` or `api_marginal` and is stored on every row, so
no downstream code can pool the two. `breakeven_narratives_per_month()` exists so the
crossover is stated rather than left for a reader to assume is anywhere in particular, and
it returns `None` rather than a number when the comparison does not apply.

The declared inputs, as of 2026-08:

| | CPU box | GPU box | API |
|---|---|---|---|
| Capital | USD 2,500 | USD 9,000 | — |
| Depreciation | 4 y straight line | 3 y straight line | — |
| Utilisation | 50% | 50% | — |
| Power / PUE | 150 W / 1.5 | 550 W / 1.5 | — |
| Electricity | USD 0.12/kWh | USD 0.12/kWh | — |
| Pricing | — | — | USD 15 / 75 per Mtok in/out |
| **Derived** | **USD 0.170/h** | **USD 0.328/h** | — |

**Two inputs move the answer more than the rest and are named as the ones to attack
first.** `utilisation`, because assuming a dedicated box runs flat out divides the hourly
cost by roughly four; and `depreciation_years`, because a longer life makes any capital
purchase look better. A test pins the utilisation relationship (0.125 costs exactly 4x
0.5) so the sensitivity cannot be quietly tuned.

**What is deliberately not claimed.** None of these figures is a claim about what any
institution actually pays. Hardware is bought at negotiated prices, electricity is
regional, API batch and cached-input discounts are not applied and would reduce the API
figures, and an institution with an existing GPU estate has already sunk the capital.

**Alternative rejected:** a single blended "cost" column. It would have been one number per
row and unarguable, which is the problem — an unarguable number built on six undeclared
assumptions is not a measurement, and the assumptions are the part a deployment reader
actually needs.

**Consequences.** The measured crossover is stated in `docs/deployability.md`: against the
frontier API at USD 22.65/1k, a USD 2,500 CPU box amortised over four years pays for
itself at roughly **2,300 narratives per month**, so a mid-size institution at 10,000
alerts/month is comfortably above it. That comparison is between B1 and B3, which differ
enormously in quality, and the document says so in the same paragraph: it is a cost
comparison and not a quality comparison. The point it supports is narrow — running locally
is not the expensive option, so cost does not stand in the way of the data-residency
argument — and it is not allowed to grow into a broader one.

---

## D-094 — The benchmark protocol: 20 discarded, 100 measured, nearest-rank, stratified twice

**Context.** "We measured latency" is not a protocol, and four of the choices inside one
are silent if made wrong.

**Decision, and why each part.**

**Warm up 20, discard them.** The first calls pay for lazy Polars kernel compilation, CUDA
context creation and the allocator's first arena. Including them reports a warm system's
latency as if it were cold — and simultaneously hides the genuine cold-start cost, which is
measured separately by `measure_cold_start` with *no* warm-up, because warm-up is the thing
cold start is defined as the absence of. A test asserts the discarded runs execute and do
not reach the result.

**Measure 100.** The floor at which a p95 is an observation rather than an extrapolation:
the 95th of 100. **p99 at n=100 is the second-worst observed run**, which is a real limit
and is pinned by a test rather than left as a caveat somebody may not read — a distribution
with one catastrophic outlier has a p99 that does not see it, and only `max_s` does. Both
are published.

**Nearest-rank percentiles, never interpolated.** An interpolated p95 is a latency that
never happened, and capacity planning sizes a queue against a latency that did. The rank is
`ceil(q*n)` on the sorted sample — a definition a reader can reproduce from the published
raw samples, which are written to `raw_samples.jsonl` for exactly that reason.

**Two draws, never pooled.** The headline distribution comes from a **seeded shuffle of the
frozen test split**, so it is representative of the corpus. The size table comes from a
**size-stratified draw**, 40 runs per band. Using one draw for both is a trap either way:
the case manifest is ordered by extraction, which correlates with size, so an in-order draw
samples the corpus's smallest cases and reports a p95 describing a workload nobody has;
and a representative draw leaves the large bands nearly empty, so the size table has
nothing in its right-hand columns. The seed is recorded in the run context.

**Both batch sizes, labelled for what they are.** Batch 1 is the interactive path an
investigator waits on. Batch 32 is the queue an overnight window is sized from — and on the
template path it is *queue* throughput, not a batched forward pass, because that path has no
batch dimension to exploit. It is labelled that way in the artifact rather than left to
imply a parallelism that is not there.

**Alternative rejected:** reporting the mean and a standard deviation. Two systems with
identical means and p99s an order of magnitude apart need different hardware, and a table
of means cannot tell them apart.

**Consequences.** The protocol is recorded in `artifacts/metrics/phase13/efficiency.json`
under `protocol`, including the draw seed, the per-band counts, the percentile method and
the exact stage list. Run-to-run p50 on this host varied 8.3–11.2 ms across five full runs
— it is a thermally throttling laptop — and that between-run spread is stated in
`docs/deployability.md` beside the within-run distribution rather than being averaged away.

---

## D-095 — End-to-end means every stage, and the graph load is cold start rather than latency

**Context.** The tempting measurement is the generation call, because it is one line and it
is the number every comparable paper reports. It would have been wrong in both directions
here.

**Decision.** The timed path is case extraction → fact extraction → serialisation →
encoding → generation → guard verification, and `EndToEndTimer.total()` sums exactly those.
`Stage.GRAPH_LOAD` is timed but **excluded from the per-narrative total**: the traversal
index is built once per process and serves every case, so charging it to each narrative
would multiply a one-off by the corpus size. It is reported as cold start, which is what it
is. `PER_NARRATIVE_STAGES` is the frozen set and a test asserts the total is the sum of the
breakdown, so the headline and the breakdown cannot disagree.

**Why it matters more than it looks.** Generation-only timing would have understated our
own systems — the fact layer and the guard are ours and they cost real milliseconds — and it
would have hidden the finding that fell out of the measurement: **case extraction is 53% of
B1's end-to-end latency and generation is 32%.** Cutting a subgraph out of a 5-million-edge
graph costs more than rendering the narrative does. That cost is paid by every row of the
matrix including the 8B arms; it does not shrink when the generator grows, it just stops
being the dominant term.

**Consequences.** Cold start is 0.64 s for the graph and 3.24 s for the index, and the
index dominates. A per-request process would pay 3.9 s every time, which would make index
construction the dominant cost of the entire system — so the service must be long-running,
and `docs/deployability.md` records that as an architectural requirement rather than a
preference.

---

## D-096 — Deployability is assessed as a software property, and the regulation is described rather than concluded

**Context.** The paper's deployment argument needs an "on-premise: yes/no" column, and the
temptation is to write the stronger sentence: that institutions *may not* send transaction
data to a third-party API.

**Decision.** `DeploymentProfile.on_premise` is decided by the executor and nothing else. It
is a statement about where the computation happens — a property of the software, true or
false independently of any jurisdiction — and it is the one column Phase 13 could fill for
all seventeen systems even though thirteen have no other measurement.

`data_leaves_perimeter` states plainly what is transmitted: for the API baselines, "the
serialised fact record for every alert: account identifiers, counterparty counts,
transaction amounts, currencies, timestamps". `regulatory_context` **describes and cites**
— the GLBA Safeguards Rule (16 CFR Part 314), the GDPR's Chapter V transfer restrictions,
internal data-governance and vendor-risk policy — and then says explicitly that whether any
particular arrangement is permissible depends on jurisdiction, contract and the
institution's own controls, and is a question for its counsel rather than for this paper.

The factual claim the paper makes is the narrow one: **the data leaves the perimeter.** The
observation that many institutions treat that as prohibited by internal policy irrespective
of what the law would permit is reported as what it is — a statement about institutional
practice, not a legal conclusion.

**Alternative rejected:** asserting that the API baselines are unavailable to regulated
institutions. It would be a stronger sentence and it would be wrong about some readers,
right about others, and unciteable for all of them. A reviewer at an applications venue who
works in the sector would notice, and a single overreach of that kind costs more credibility
than the sentence buys.

**Consequences.** `docs/deployability.md` opens by saying what it is not. The summary table
carries the on-premise column for all seventeen rows and leaves every other column empty
where nothing was measured, which is the right shape: **twelve of seventeen systems are
on-premise-capable by construction**, and that is knowable without a GPU.

---

## D-097 — Phase 13 measured what exists, and thirteen rows are absences with named blockers

**Context.** The Phase 13 brief opens by asking for confirmation that Phase 11 is complete —
all systems trained and evaluated, checkpoints available. It is not, and the repository has
said so since D-087: `aggregate.json` reads `n_rows: 0`, all 25 declared runs are `pending`,
and `artifacts/checkpoints/` holds Phase 7 encoder arms and nothing else.

**Decision.** Build the instrument in full to the protocol, measure every system that can be
measured on this machine, and write every system that cannot as a row carrying its blocker —
the same shape Phases 11 and 12 already use. `SystemEfficiency.__post_init__` **refuses to
construct an unmeasured row without a blocker**, so the invariant is enforced at the type
rather than by anyone remembering.

**What is measured, and it is not a proxy.** B1 and B2 are complete systems that run end to
end on this hardware, at n=100 with 20 discarded. Beyond them, the *components* every other
row shares were measured and recorded in `components.json`: the graph load and index build,
the encoder's inference and training footprint, and the guard's verification pass. On the
day a card arrives, the only unmeasured stage is the decoder.

**What is not measured, and is not estimated either.** No latency, throughput or
operational-cost figure is given for the 8B arms. The minimum-hardware figures for them
(16 GB inference, 24 GB training) are arithmetic on the model, are stated as such in the
same sentence, and are the only extrapolation anywhere in the phase. The API rows carry a
cost estimate from published pricing and *measured* token counts from this corpus, labelled
`api_marginal` and never presented as a measurement of those systems.

**The guard figure carries the caveat that would otherwise be misread.** What was measured
is the guard's **verification** pass — 4.11 ms for four candidates, model-independent, which
is why it is measurable on a machine that cannot run the model. The four *generations* the
guard requests are not included. On B1 that yields 1.25x, which is a fact about B1 and is
stated in the table caption, in the deployability document and in the module as something
that must never be quoted as the guard's overhead in general.

**Consequences.** The efficiency table is 2 of 17 rows measured, and `EfficiencyTable.coverage()`
prints that ratio into the LaTeX caption so "the efficiency table" is never mistaken for a
complete one. The frontier figure has one honest point — B1, whose Zero-Hallucination Rate
is the only faithfulness measurement in the project — and its title says "16 of 17 systems
unmeasured". Every one of those absences is recoverable by rerunning `make benchmark` on a
machine with a 24 GB card once Phase 11 has run; nothing here needs rebuilding.

---

## D-098

**The narrative corpus and the fact records ship under CDLA-Sharing-1.0, not Apache-2.0,
and the release is two licence-homogeneous bundles that must never be merged.**

*Phase 14, 2026-08-06.*

**The tempting reading, and why it is wrong.** CDLA-Sharing-1.0 §3.5 exempts "Results" —
outputs of computational use — from the share-alike obligation, and the Phase 1 data card
recorded that trained models, generated narratives and metrics are Results. "Generated
narratives" appears in that list, so the obvious move at release time was to put the whole
corpus in the Apache-2.0 bundle.

**What the corpus actually contains.** A Bronze record's `target_narrative` reads:

> *"This referral describes activity centred on account 021611|800F41B10 ... The
> transactions fall between 2022-09-07 13:30 and 2022-09-08 08:52 ... Aggregate inflow to
> the subject account was around 6,273 US Dollar."*

Account identifiers, timestamps, currencies and amounts, verbatim from the source. And
every record embeds its full `case_facts`, which carries `entity_inventory.node_ids`,
`focal_entity.first_seen` and `flow.max_single_transfer` — individual transaction values,
not aggregates over them. §3.5 defines Results as excluding *more than de-minimis portions
of the original Data*. Fifteen thousand narratives each naming real source identifiers and
real source values is not de minimis.

**Decision.** The corpus, the fact records and the case store are **Enhanced Data** and
ship under **CDLA-Sharing-1.0**, with the agreement text, attribution to IBM as Data
Provider, and the §3.2 record of changes. Code, model weights, metrics, figures and
`RESULTS.md` remain **Apache-2.0** as genuine Results — a checkpoint is a tensor of
weights and embeds no source row.

**Why this costs nothing.** CDLA-Sharing-1.0 *permits* redistribution. The conservative
reading loses no reach; it only obliges a downstream user to keep the same terms. The
unconservative reading, if wrong, is a licence breach in a paper's artifact release. The
asymmetry is total, and this is the second time in this project that the cheaper-if-wrong
direction has been the right one to take (cf. D-025 on typed absence).

**Enforcement, because a comment is not a mechanism.** `scripts/14_package_release.py`
declares the two bundles as data and writes each with its own licence text and NOTICE.
`tests/unit/test_release_packaging.py` asserts that no `schemas/` path enters the CDLA
bundle, that no `data/` path enters the Apache bundle, that nothing Elliptic2-derived
enters either, and that the CDLA bundle carries both the attribution and the change record.
`scripts/14_verify_release.py` asserts `tests/fixtures/NOTICE` still names both
redistributions — the fixture NOTICE is a licence obligation, and deleting it is a breach
rather than a tidy-up.

**The quickstart fixture is the same question in miniature.** The committed 220-record
`bronze_quickstart.jsonl.gz` is a redistribution of Enhanced Data by the same reasoning, so
`tests/fixtures/NOTICE` was extended to cover it with its own §3.2 change record. The golden
file beside it holds *metrics computed over* the fixture, which is a Result, and stays
Apache-2.0. Two files in the same test tree under two licences is the correct answer here,
not an inconsistency.

---

## D-099

**Nothing Elliptic2-derived is released, because nothing exists — and the reconstruction
script ships anyway.**

*Phase 14, 2026-08-06.*

The Phase 1 card recorded Elliptic2 as `redistributable=False` with its data licence
**unlocated**, and flagged the question as one that *"must be resolved before Phase 14"*.
It was not resolved, for the simplest possible reason: **access was never requested, so the
substrate has never been ingested.** There are zero Elliptic2 cases, zero fact records and
zero narratives anywhere in this repository.

The standard release pattern for a gated corpus — publish case IDs, fact records and
narratives, plus a script that rebuilds the graphs from the user's own licensed copy —
therefore has nothing to publish. All three inputs are empty.

**Decision.** Ship `scripts/14_reconstruct_elliptic2.py` regardless, and have it say so.

Two reasons. First, **a release that silently omits Elliptic2 implies a second-substrate
half that a reader may assume exists**; a script that prints "Elliptic2 access was never
requested by this project and the substrate has never been ingested" cannot be misread.
Second, the path is then in place, tested and documented on the day access arrives, rather
than being written under submission pressure.

**No Elliptic2 checksum is pinned anywhere**, and a test asserts the string `sha256` does
not appear in that script. We have never seen the files; a pinned digest would be a
fabrication, and a fabricated digest in a verification path is worse than none because it
would fail a legitimate copy.

**What a future session must do before publishing anything Elliptic2-derived:** read the
agreement actually presented with the access grant, then update `docs/data_cards/elliptic2.md`
§2, the registry's `licence` and `redistributable` fields, and this file. Redistribution is
a separate question from whether derived artifacts may be published — even a fully closed
dataset usually permits publishing metrics — and that must be confirmed against the terms
held, not assumed.

---

## D-100

**The reproduction tolerance policy: deterministic stages reproduce exactly; stochastic
stages reproduce their conclusions, not their digits.**

*Phase 14, 2026-08-06.*

A reproducibility claim that is not bounded is not a claim. "Reproducible" without a
tolerance invites a reader to diff the fourth decimal of an AUC-PR and conclude the work
does not replicate, or — worse — invites us to quietly widen the band whenever a rerun
disagrees.

**The policy**, tabulated in full in `docs/REPRODUCTION.md` §6:

| Stage | Tolerance |
|---|---|
| Ingestion, splits, facts, Bronze | **Exact.** Byte-identical, content hashes must match. |
| Bronze evaluation rates | **Exact**; ±0.0001 on bootstrap CI bounds |
| Encoder AUC-PR, per seed | **±0.02** absolute |
| Encoder AUC-PR, 3-seed mean | **±0.01** absolute |
| **The encoder gate outcome** | **Must reproduce exactly** |
| **Ablation signs** (edge features positive; PE and focal nulls) | **Must reproduce** |
| Latency p50/p95/p99 | **±2×** on a laptop, ±20% on a dedicated idle host |
| VRAM, parameter counts, disk size | **Exact** |
| Cost estimates | **Not reproducible** — list prices at 2026-08 |

**The reasoning.** Splitting the policy at "deterministic versus stochastic" is the wrong
cut, because it puts the *gate outcome* — a comparison, not a measurement — on the loose
side of the line. The right cut is between **numbers** and **conclusions**. A per-seed
AUC-PR is a number and gets a band. "GATv2 beats the MLP control at every seed" is the
claim the paper makes, and it gets **no** band: if it flips on a rerun, that is a finding
about the method, not a tolerance breach, and treating it as one would be the specific
dishonesty this policy exists to prevent.

**Named sources of non-determinism**, rather than a wave at "GPU variance": scatter/gather
atomics summing in nondeterministic order (the dominant source, and unfixable by seeding);
cuDNN algorithm selection varying with card, driver and free memory; **TF32 on Ampere and
later** — the development host is Turing and does *not* use it, so an A100 or 4090 will
produce visibly different digits from the published table for that reason alone; and
`torch.use_deterministic_algorithms` deliberately **not** enabled, because several PyG
scatter ops have no deterministic implementation and enabling it raises rather than slows.
Making the encoder deterministic would mean changing the model.

**The 2× latency band is measured, not chosen for comfort.** Phase 13 observed B1's p50
varying 5.4–11.2 ms across seven full protocol runs on one afternoon on this laptop, and no
Phase 13 conclusion rests on a smaller difference (D-097). The band is that measurement,
carried forward.

---

## D-101

**The quickstart runs the real evaluation harness over a committed fixture, and asserts it
exactly.**

*Phase 14, 2026-08-06.*

The venue treats "code available on request" as a desk-reject trigger, so the repository
has to work for a stranger. The binding constraint: `bronze.jsonl` is 232 MB and gitignored,
and the raw AMLworld release is a ~20 GB manual Kaggle download behind an API token. A
quickstart that begins with either is not a quickstart, and one that fails is worse than
none because it signals the rest is unreliable too.

**Decision.** Commit a **220-record stratified slice** — 20 from each of the eleven
narrative families, taken in sorted `case_id` order so the draw carries no decision — and
have `scripts/14_quickstart.py` stage it and invoke **`scripts/10_evaluate.py` itself**,
overriding only `paths.processed_dir`, `paths.metrics_dir` and `paths.runs_dir`. Runtime:
**1.5 s**. Whole path from a clean clone: under a minute, install included.

**Three choices inside that, each load-bearing.**

*It calls the real harness, not a reimplementation.* A bespoke scorer would be a second
implementation of the thing invariant 1 exists to protect, and it would agree with the
first on their shared misreading — the same argument as D-034 and the reason `tests/oracle.py`
exists. The vocabulary, the schemas and the frozen checker policy all come from the
repository unchanged.

*The assertion is exact, not toleranced.* Bronze renders deterministically from the fact
record, so the fixture's scores are bit-reproducible anywhere. A tolerance would hide
exactly the class of bug this check exists to catch. It is the one place in the project
where "matches to every digit" is the right demand, and D-100 says why it is the only one.

*It is compressed.* 220 records is 3.3 MB raw and 244 KB gzipped — under the 512 KB
pre-commit large-file threshold, which is a guard worth keeping rather than raising. The
alternative was ~24 uncompressed records, too few to cover eleven families.

**The fixture's numbers deliberately differ from the corpus's** — Fact Coverage 0.8359
against 0.8595, H9 0.45 against 0.9179 — because the families are weighted evenly rather
than as they occur. That divergence is stated in the script's docstring, in the README, in
`tests/fixtures/NOTICE` and in `docs/REPRODUCTION.md`, in each case beside the number, so
that nobody quotes a fixture figure as a corpus figure.

---

## D-102

**Release verification runs against a `git archive` export, on a schedule, and reports all
nine checks rather than aborting on the first.**

*Phase 14, 2026-08-06.*

**Why an export and not the working tree.** The whole failure mode being defended against
is "works on my machine": an untracked file, a populated `data/`, a stale `.venv`, a
checkpoint that happens to be on disk. `scripts/14_verify_release.py` stages everything
into a **scratch git index** (`GIT_INDEX_FILE`, so the real index is never touched),
writes a tree, and `git archive`s it into a pristine directory. What the verification sees
is exactly what a clone delivers.

**Why nine checks and not one.** A release failing one check and passing eight is a
different situation from one failing all nine, and a script that aborts on the first cannot
tell you which. Every check reports pass, fail or skipped-with-reason, and the JSON report
carries all three counts — so a green run with six skips can never read as a clean one.

**Why the secret scan is over the object database.** Scanning the working tree is not
enough: **a credential committed and later removed is still in every clone.** The scan runs
`gitleaks detect` over the full history with `.gitleaks.toml`, whose every allowlist entry
is a documented false positive with its reason. It is also the one check that still runs
when the clean clone could not be built, because it does not need the clone.

**Why on a schedule.** The answer decays without anyone touching the repository — a base
image is rebuilt, a wheel is yanked, an index moves. **The manuscript names the URL**, so a
release verified in August and broken in October is broken. `verify-release.yml` runs
weekly, on every `v*` tag, and on any change to the packaging, the images or the lockfile.

**Building and running the artifact found five defects that reading it could not.** This is
the argument for the phase, and it is worth stating concretely rather than as a principle:

| Defect | Why inspection missed it |
|---|---|
| `07c_report_tables.py --help` exited 1 | It works on a host that has run `make train-encoder` |
| **uv pinned at 0.4.20 predates `--group`** — the image *and* `ci.yml` had never run | The repository had no remote, so CI had never executed once |
| **No `.dockerignore`** — an ~8 GB build context | `.gitignore` looks like it covers this, and does not apply to `docker build` |
| **Four test modules fail at *collection* without torch**, taking `make smoke` down | The development host has the GPU extras installed |
| **15 statistics tests need scipy**, which the light environment does not install | Hidden behind the collection errors above until those were fixed |
| **`uv.lock` held matplotlib 3.11.1 against a 3.9.2 pin** | Only surfaced by relocking for the new `stats` extra |
| **`build_optimizer` picks `PagedAdamW8bit` over CPU tensors**, failing at the first `.step()` | `bitsandbytes` had never been installed in the development environment |
| Two of the nine release checks were themselves wrong | They had never been run against a real clean clone |

The fourth is the worst of them: **`make smoke` is documented as the CI gate and as the
second command a stranger types, and it did not work in the environment `make install`
produces.** The fifth compounded it, and its fix is a **design decision rather
than a patch**: `eval/statistics.py`, `eval/report.py` and `experiments/aggregate.py` are
measurement code under invariant 1, so guarding them behind `importorskip` would leave the
CI gate not gating the statistics — while installing the `eval` extra to satisfy them would
drag torch in via `bert-score` and end the light environment. A new **`stats` extra**
(scipy, statsmodels, krippendorff, no torch) is what makes both properties hold at once. Every sibling GPU test module already carried `pytest.importorskip("torch")`;
these four had been written without it. A collection error is not a skip — it aborts the
whole run — so the failure mode was total rather than partial.

**One check exists because of a bug it found.** `script-help` requires every `scripts/*.py`
to answer `--help` with exit 0. `07c_report_tables.py` did not: it checked for
`encoder_report.json` before parsing arguments, so on a clean clone the only way to ask what
it did was to be told the file was missing. Undiscoverable to exactly the reader this phase
is for.

**And that check had to learn a distinction on its first real run.** The four GPU
entrypoints also exited 1, because they import torch — which `make install` deliberately
does not provide, since phases 1-6 and 10 are CPU-only by design and torch lives behind the
`graph`/`llm` extras (CLAUDE.md §4). Failing the release for that would be demanding the
light environment stop being light, and loosening the check to "warn on any failure" would
have thrown away the `07c` catch. The rule is therefore **specific**: a `--help` failure
whose traceback names a module on the optional-extra allowlist is counted and reported
separately; **every other non-zero exit still fails the release.** The module name is parsed
out of the traceback and matched against a closed list, never inferred from the message
text, so a script that breaks for an unrelated reason cannot disguise itself as a missing
extra.

**`no-data-committed` had the same shape of error in the other direction**: it walked the
`.venv` that the `install` check had just created inside the clone and reported polars'
100 MB shared object and pyarrow's test `.parquet` fixtures as leaked artifacts. A check
firing at its own side effect trains its reader to ignore it, which is worse than not having
it. Tool-generated directories are now excluded from the walk by an explicit list, so what
the check reports on is what git would actually deliver.
