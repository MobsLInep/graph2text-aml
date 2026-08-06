# Deployability

**What this document is.** A factual, per-system assessment of whether each row of the
Phase 11 matrix can run inside a financial institution's own network, what hardware it
needs, what it costs at a realistic alert volume, and what data — if any — leaves the
institution's control.

**What this document is not.** It is not a legal opinion. Whether a given arrangement is
permissible for a given institution depends on its jurisdiction, its contracts, its
regulator and its own controls, and that is a question for its counsel. What is stated here
is what the *software* does with data. Where the regulatory landscape is described, it is
described and cited, not concluded from.

Generated from `artifacts/metrics/phase13/efficiency.json`. Every measured number below was
measured; every unmeasured one says so.

---

## 1. The hardware everything below was measured on

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2050, 4 GB VRAM |
| Driver / CUDA | 595.84 / 12.1 |
| torch | 2.4.0+cu121 |
| CPU | 12th Gen Intel Core i5-12450H, 12 logical cores |
| System RAM | 7 GB |
| OS / Python | Linux 7.0.0-28-generic / 3.11.14 |

**This is a laptop, and it is a binding constraint rather than a footnote.** Llama-3.1-8B at
nf4 is 4.5–5.6 GB of weights before a single activation — more than this card holds — and
CPU offload is closed by the 7 GB of system RAM (D-068). That is why thirteen of the
seventeen rows below are unmeasured, and it is why the two that are measured are measured
properly rather than approximated.

A second consequence, stated because it affects how every number below should be read: a
laptop throttles, and it is sensitive to whatever else is running. Run-to-run p50 for B1
varied **5.4–11.2 ms across six full protocol runs** on the same afternoon — a 2x
between-run spread, larger than the difference the case-size bands resolve. One run taken
while the test suite was executing concurrently showed p95 inflated to 30 ms and index
build inflated threefold; it was discarded and rerun on an idle machine. The figures below
are that final idle run.

**Nothing in this document rests on a difference smaller than 2x**, and a server-class host
would be considerably steadier. Read the orders of magnitude, not the milliseconds.

---

## 2. The measured systems

### B1 — deterministic template (the faithfulness ceiling)

| | |
|---|---|
| Runs entirely on-premise | **Yes** |
| Data leaving the perimeter | **Nothing** |
| Minimum viable hardware | A commodity CPU server. No accelerator. |
| Learned parameters | 0 |
| End-to-end latency (p50 / p95 / p99 / max) | 5.4 / 22.1 / 31.5 / 61.4 ms, n = 100 |
| Throughput, batch 1 | 123.9 narratives/s |
| Throughput, 32-case queue | 206.7 narratives/s |
| Cold start | 0.92 s (graph load); 7.3 s including the traversal index |
| Peak host RAM | 3.08 GB |
| Peak VRAM | n/a — no accelerator is touched |

Stage breakdown, mean per narrative:

| Stage | ms | Share |
|---|---:|---:|
| Case extraction | 5.82 | 72% |
| Fact extraction | 0.72 | 9% |
| Serialisation | 0.09 | 1% |
| Generation (template render) | 1.44 | 18% |
| *Guard verification, 4 candidates* | *1.90* | *(added by the guarded variant)* |

**The finding worth carrying into the paper: nearly three quarters of the end-to-end
latency of the template system is case extraction, not generation.** Cutting a subgraph out of a
5-million-edge graph costs more than rendering the narrative does. That is a cost every row
of the matrix pays, including the 8B arms, and it does not shrink when the generator gets
bigger — it just stops being the dominant term.

### B2 — Phase 7 GATv2 classifier + template

| | |
|---|---|
| Runs entirely on-premise | **Yes** |
| Data leaving the perimeter | **Nothing** |
| Minimum viable hardware | A commodity CPU server. The accelerator buys latency, not feasibility. |
| Total / trainable parameters | 628,058 / 628,058 |
| Model size on disk | 2.53 MB (checkpoint, including the fitted feature space and optimiser state) |
| End-to-end latency (p50 / p95 / p99 / max) | 15.8 / 24.5 / 28.9 / 39.1 ms, n = 100 |
| Throughput, batch 1 | 58.9 narratives/s |
| Cold start | 7.65 s (graph + index + encoder load) |
| Peak VRAM, inference | **0.025 GB reserved** (0.012 GB allocated) |
| Peak VRAM, encoder training | 0.055 GB reserved (0.047 GB allocated), batch 32, 18.1 ms/step |

Stage breakdown, mean per narrative:

| Stage | ms | Share |
|---|---:|---:|
| Case extraction | 2.86 | 17% |
| Fact extraction | 0.74 | 4% |
| Serialisation | 0.10 | 1% |
| **Encoding (GATv2 forward)** | **11.76** | **69%** |
| Generation (template render) | 1.52 | 9% |

The encoder dominates, and at 628 K parameters on a 4 GB card that is a launch-overhead
story rather than a compute story: the graphs are small (median 12 nodes), so most of the
11.8 ms is kernel launch and host-device transfer, not arithmetic. A batched encoder forward
would amortise nearly all of it. This is measured as it is deployed — one case at a time,
the interactive path — and the batched path is not measured.

**The whole system needs 25 MB of device memory.** That number matters for the deployment
argument: the graph half of this architecture is not what makes it expensive.

---

## 3. Latency by case size

A 150-node case does not cost what a 20-node case costs, and a single mean over the corpus
describes neither. Measured over a size-stratified draw, 40 runs per band:

| Band (nodes) | B1 p50 / p95 (ms) | B2 p50 / p95 (ms) |
|---|---:|---:|
| 0–24 | 3.3 / 5.0 | 18.5 / 30.2 |
| 25–49 | 8.9 / 16.1 | 21.7 / 32.2 |
| 50–99 | 7.6 / 19.5 | 25.9 / 36.8 |
| 100+ | *no case in this corpus* | *no case in this corpus* |

The size effect is roughly 2–4× from the smallest band to the largest, and it is **monotone
in p95 for both systems**. B1's p50 is not monotone — 7.6 ms in the top band against 8.9 ms
in the middle one — and at 40 runs per band against a 2× between-run spread that inversion
is noise rather than a finding. Size the queue from the p95 column.

**The 100+ band is empty**: the Phase 2 node budget is 150, but no case in the frozen test
split reaches 100 nodes. A deployment reader sizing for large cases should treat the 50–99
row as the top of the measured range and not extrapolate the trend beyond it.

---

## 4. What the guard costs

The Phase 9 inference guard samples four candidates, verifies each against the fact record,
selects the best, and repairs once if the best still contradicts.

| | B1, guard off | B1, guard on |
|---|---:|---:|
| p50 | 5.4 ms | 5.7 ms |
| p95 | 22.1 ms | 9.8 ms |
| Mean | 8.1 ms | — |
| **Guard stage, mean** | — | **1.90 ms** |

**Take the guard's cost from the stage mean, not from differencing the two columns.** The
between-run spread on this host is larger than the effect, which is why guard-on p95 reads
*lower* than guard-off p95 above — that is measurement noise between two independently
drawn runs, not a guard that makes the system faster. The 1.90 ms stage figure is measured
inside each guarded narrative and is the number to use.

**Read this figure carefully, because the obvious reading is wrong.** What is measured is
the guard's *verification* pass: claim extraction plus the Phase 3 checker, run once per
candidate, plus selection. It is 1.90 ms for four candidates — about 0.5 ms per
candidate, and it does not depend on which model produced the text, which is precisely why
it is measurable on a machine that cannot run the model.

What is **not** measured is the four *generations* the guard requests. On B1 a "generation"
is a 1.4 ms template render, so the guard's full cost there is about 4.3 ms of extra
rendering plus 1.9 ms of verification against an 8.1 ms mean. On a system whose generation
is an 8B decoder, the guard will cost roughly **four generations plus 2 ms**, and the
generation term will dominate completely. Any ratio computed from B1 is a fact about B1 and
must never be quoted as the guard's overhead in general.

Repair was disabled for this measurement. Bronze is faithful by construction, so repair
would never fire, and leaving it on would report a repair rate of zero as though it were a
property of the guard rather than of the input.

---

## 5. The unmeasured systems, and what is known about them

Thirteen rows could not be measured. Each is listed with its blocker rather than omitted
(invariant 7).

### The local 8B arms — B6, B7, B8, S1, S2, A1, A2, A3_F3, A3_F4, A4, A5, A6

| | |
|---|---|
| Runs entirely on-premise | **Yes** — this is the point of the architecture |
| Data leaving the perimeter | **Nothing** |
| Blocker | No trained checkpoint (Phase 11 has not run, Gate 8 is open) and no accelerator that fits the model |
| Minimum viable hardware, **stated not measured** | 16 GB accelerator for inference at nf4; 24 GB to train under QLoRA at the Phase 9 sequence length |
| Latency, throughput, VRAM, cost at volume | **Not measured. Not estimated.** |

The minimum-hardware figures are arithmetic on the model, not measurements: 8.03 B
parameters at 4-bit is ≈ 4.5 GB of weights, ≈ 5.6 GB with the unquantised layers and
embeddings, plus KV cache and activations. They are stated because a reader sizing hardware
needs a starting point, and they are labelled because they are not measurements.

**No latency, throughput or operational-cost figure is given for these rows, and none
should be inferred.** A deployment reader wants to know whether an 8B system answers in one
second or thirty, and this project does not yet know. What *is* known and measured is that
everything around the decoder — case extraction, the fact layer, serialisation, the encoder,
the guard's verification — totals **under 20 ms**, so whatever the decoder costs, it is
essentially the whole per-narrative budget.

What this project ships on top of the base model is small: the encoder is 2.5 MB, and the
LoRA adapter and fusion projector would add tens of megabytes. The 5.6 GB base model is a
public download the institution hosts once. That asymmetry — a multi-gigabyte commodity
backbone plus a few tens of megabytes of contribution — is itself a deployment argument.

### The frontier-API baselines — B3, B4, B5

| | |
|---|---|
| Runs entirely on-premise | **No** |
| Data leaving the perimeter | **The serialised fact record for every alert**: account identifiers, counterparty counts, transaction amounts, currencies, timestamps and the derived risk signals |
| Blocker on measurement | No API credentials; zero calls have been made |
| Minimum viable hardware | None. A network egress path and a vendor contract. |
| Estimated cost per 1,000 narratives | B3, B4: **USD 22.65**; B5: **USD 67.94** |

The cost figures are estimates, and their inputs are stated in §6. The token counts behind
them are measured from this corpus (mean 569 prompt tokens, 188 completion tokens per
narrative); the prices are published list prices as of 2026-08 and will move. B5 is charged
for three calls per narrative because it generates, self-verifies and repairs, and is
deliberately given more inference compute than any of our arms (D-084) — charging it for one
call would price it as something it is not.

**On the regulatory position.** Sending customer transaction records to a third-party
endpoint puts them outside the institution's direct control. Institutions weigh this
against the GLBA Safeguards Rule (16 CFR Part 314) in the United States, the GDPR's Chapter
V restrictions on transfers outside the EEA, and their own internal data-governance and
vendor-risk policy. Many institutions treat outbound transmission of customer transaction
data as prohibited by internal policy irrespective of what the law would permit. Whether any
particular arrangement is permissible is a question for the institution's counsel and is not
answered here. What is factual, and what the table records, is that the data leaves the
perimeter.

---

## 6. Cost at a realistic volume: 10,000 alerts per month

**The amortisation assumptions, stated before the numbers, because comparing an amortised
local cost against a marginal API price without stating them is the standard way this
comparison misleads.**

| Input | Local (CPU box) | Local (GPU box) | API |
|---|---|---|---|
| Capital | USD 2,500 | USD 9,000 | — |
| Depreciation | 4 years, straight line | 3 years, straight line | — |
| Utilisation | 50% | 50% | — |
| Power draw | 150 W | 550 W | — |
| PUE | 1.5 | 1.5 | — |
| Electricity | USD 0.12/kWh | USD 0.12/kWh | — |
| Pricing | — | — | USD 15 / 75 per M input / output tokens |
| **Derived hourly cost** | **USD 0.170** | **USD 0.328** | — |

None of these is a claim about what any institution actually pays. Hardware is bought at
negotiated prices, electricity is regional, batch and cached-input API discounts are not
applied, and an institution with an existing GPU estate has already sunk the capital.
Substitute your own figures; the model is in `CostAssumptions` and shows its working.

**The two costs are different kinds of number.** An API price is marginal — one more
narrative costs one more narrative's tokens, and zero narratives cost nothing. A local cost
is capital already spent, and at low volume it is dominated entirely by the box existing.

At 10,000 alerts/month:

| System | Compute time consumed | Marginal cost | Total monthly cost |
|---|---:|---:|---|
| B1 | 81 seconds | USD 0.004 | **≈ USD 52** — the box, essentially idle |
| B2 | 170 seconds | USD 0.008 | **≈ USD 52** — the same box |
| B3, B4 | n/a | USD 226.45 | **USD 226** |
| B5 | n/a | USD 679.36 | **USD 679** |
| The 8B arms | **unmeasured** | **unmeasured** | **not computable** |

The local systems consume under five minutes of compute per month at this volume. Their
cost is the cost of owning a server, and the per-narrative figure is an artefact of dividing
a fixed cost by a small number.

**Breakeven.** Against the frontier API at USD 22.65 per thousand, a USD 2,500 CPU box
amortised over four years pays for itself at roughly **2,300 narratives per month**. Below
that volume the API is cheaper; above it the local system is, and the gap widens linearly.
A mid-size institution at 10,000 alerts/month is comfortably above the crossover.

**This is a cost comparison and not a quality comparison.** B1 is a deterministic template
and B3 is a frontier model; they are not interchangeable, and nothing here says the cheaper
one is the better one. The point of the crossover figure is narrower and it is this: at
realistic institutional volume, running locally is not the expensive option. The reason to
run locally is that the data does not leave — the cost argument merely removes the usual
objection to it.

---

## 7. What a mid-size institution would actually need

For the template and classifier systems (B1, B2), measured: **one commodity server, no
accelerator.** 8 cores and 16 GB RAM is ample. Peak host RAM was 3.1 GB, and that figure
is dominated by holding the 5-million-edge graph and its traversal index in memory — which
is a one-time, shared cost, not a per-request one. A second process would not double it if
the index were shared.

For the 8B arms, **stated and not measured**: one 16 GB accelerator for inference, one 24 GB
accelerator to train. A single such card sits inside an ordinary rack server and inside an
ordinary capital-approval threshold. Whether one card is enough for a given alert volume
depends on the decoder throughput, which this project has not measured.

Two operational points that hold for every local row:

- **Cold start is 0.9–7.7 s**, dominated by building the traversal index over the substrate
  graph (6.0 s in the final run, 3.2–6.0 s across runs). A long-running service pays this
  once. A per-request process would pay it every time — three orders of magnitude more than
  the 5.4 ms it takes to serve a narrative — which would make index construction the
  dominant cost of the entire system. **The service must therefore be long-running, and that
  is an architectural requirement rather than a preference.**
- **Nothing in the measured path needs network access at inference time.** The vocabulary,
  the split manifests, the fact schema and the encoder checkpoint are all local files.

---

## 8. Summary table

| System | On-prem | Data leaves | p50 latency | VRAM | USD/1k | Status |
|---|:---:|---|---:|---:|---:|---|
| B1 | ✓ | Nothing | 5.4 ms | — | 0.00038 | **measured** |
| B2 | ✓ | Nothing | 15.8 ms | 0.025 GB | 0.00080 | **measured** |
| B3 | ✗ | Full fact record | — | — | 22.65 | cost estimated; no credentials |
| B4 | ✗ | Full fact record | — | — | 22.65 | cost estimated; no credentials |
| B5 | ✗ | Full fact record | — | — | 67.94 | cost estimated; no credentials |
| B6 | ✓ | Nothing | — | — | — | no accelerator (D-068) |
| B7 | ✓ | Nothing | — | — | — | untrained; no accelerator |
| B8 | ✓ | Nothing | — | — | — | untrained; no accelerator |
| S1 | ✓ | Nothing | — | — | — | untrained; Gate 8 open |
| S2 | ✓ | Nothing | — | — | — | untrained; Gate 8 open |
| A1 | ✓ | Nothing | — | — | — | untrained; Gate 8 open |
| A2 | ✓ | Nothing | — | — | — | untrained; no accelerator |
| A3_F3 | ✓ | Nothing | — | — | — | untrained; no accelerator |
| A3_F4 | ✓ | Nothing | — | — | — | untrained; no accelerator |
| A4 | ✓ | Nothing | — | — | — | untrained; no accelerator |
| A5 | ✓ | Nothing | — | — | — | untrained; no accelerator |
| A6 | ✓ | Nothing | — | — | — | untrained; no accelerator |

**Twelve of seventeen rows are on-premise-capable by construction, and that column is
complete even where every other column is empty** — it is a property of where the
computation happens, not of how fast it happens or how well it scores. That is the column
the paper's deployment argument turns on, and it is the one column Phase 13 could fill for
every system.
