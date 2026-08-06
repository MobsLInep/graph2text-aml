# Model card — GATv2 case encoder

**The primary graph encoder.** Scores a transaction subgraph for suspicion and emits the
pooled token sequence the (untrained) fusion layer is designed to consume.

| | |
|---|---|
| Card written | 2026-08-06 (Phase 14), for release `v0.1.0` |
| Model type | Graph attention network (GATv2), 3 layers, 8 heads, attention pooling |
| Parameters | **628,058**, all trainable |
| Size on disk | **2.53 MB** per checkpoint |
| Checkpoints | `gatv2_seed42.pt`, `gatv2_seed43.pt`, `gatv2_seed44.pt` |
| Framework | PyTorch 2.4.0 + PyTorch Geometric 2.6.1, CUDA 12.1 |
| Licence | Apache-2.0 (a *Result* under CDLA-Sharing-1.0 §3.5) |
| Code | `src/g2t_aml/models/encoder/` |
| Trained by | `make train-encoder` (`scripts/07_train_encoder.py experiment=encoder_sweep`) |

---

## 1. Intended use

**Primary use.** Produce (a) a case-level suspicion score and (b) a 16-token pooled
representation of a transaction subgraph, as the graph half of the Graph2Text AML
architecture. The score is written back into every fact record as
`model_signal.gnn_risk_score`; the tokens are what the fusion layer projects into a
language model's embedding space.

**Secondary use.** As a baseline graph classifier for AML subgraph suspicion on AMLworld,
and as the reference point against which the MLP control is read.

**Users.** Researchers reproducing or extending this work. Not an end-user component.

### Out of scope

- **Any deployment decision about a real account or person.** See §6 and
  `docs/ETHICS.md` §2.
- **Any substrate other than AMLworld HI-Small.** The feature space is fitted to this
  substrate: 15 currencies, 7 payment formats, per-currency amount statistics, all baked
  into the checkpoint's `feature_space` block. Applying it elsewhere is out of
  distribution and the checkpoint will not tell you so.
- **Standalone explanation.** The attention weights are not an explanation of a decision;
  §4.4 reports what they do and do not align with.

---

## 2. Architecture and training

### Architecture

```
encoder_config:
  arch: gatv2          hidden_dim: 256      layers: 3        heads: 8 (concat)
  dropout: 0.2         edge_dim: 16         residual: true   norm: layernorm
  readout: attention_pool                   n_pooled_tokens: 16
  typology_head: true  use_edge_features: true
```

Six arms sit behind one `BaseEncoder` and differ **only** in `message_passing`. That is a
deliberate constraint: it is what makes the arm comparison in §4.2 a comparison of message
passing rather than of six differently-tuned models.

### Training configuration

```
optimizer: adamw     lr: 1e-3            weight_decay: 1e-4      scheduler: cosine
epochs: 100          early_stop: val_auc_pr, patience 15
batch_size: 64       eval_batch_size: 256
loss: focal (gamma 2.0), class_weights: inverse_freq, typology_weight: 0.3
grad_clip: 1.0       lap_pe_dim: 8       rw_pe_dim: 16    lap_pe_sign_flip: true
seeds: [42, 43, 44]  n_bootstrap: 2000   ci: 0.95
```

Seed 42 selected at **epoch 31** on val AUC-PR 0.8369. Hardware: 1 × RTX 2050 (4 GB); the
whole 9-arm × 3-seed sweep is ~6.5 h wall-clock (`docs/ETHICS.md` §6).

### Training data

| | |
|---|---|
| Substrate | IBM AMLworld HI-Small — **synthetic**, 515,088 accounts, 5,078,345 transactions |
| Cases | 10,932 train / 2,028 val / 3,196 test, **temporally disjoint and frozen** |
| Split manifests | `schemas/splits/amlworld/`, committed ID lists with content hashes (invariant 2) |
| Positive rate | Laundering rate in the source graph is 1 in 981; cases are stratified, not uniformly sampled |
| Label | Case-level suspicion, plus an auxiliary typology head over 8 laundering typologies |

Full provenance and licence in [`../data_cards/amlworld_hi_small.md`](../data_cards/amlworld_hi_small.md)
and [`../dataset_cards/case_facts.md`](../dataset_cards/case_facts.md).

**Every node feature is recomputed from the case's own edges.** The interim node table's
`in_degree` / `out_degree` / `degree` / `total_received` / `total_sent` columns are global
aggregates over the whole 515,088-account graph, computed across both sides of the temporal
boundary — reading them would leak a test-window account's training-window activity into
its encoding. A test overwrites all five columns with absurd constants and requires the
feature tensor to be unchanged (D-059).

---

## 3. How to use

```python
import torch
from g2t_aml.models.encoder import build_encoder   # see src/g2t_aml/models/encoder/

ckpt = torch.load("artifacts/checkpoints/encoder/gatv2/gatv2_seed42.pt", map_location="cpu")
model = build_encoder(ckpt["encoder_config"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

`ckpt["feature_space"]` carries the currency and payment-format vocabularies and the
per-currency amount statistics the features were standardised with. **Featurising a case
with any other feature space silently produces garbage** — it is stored in the checkpoint
for exactly that reason.

To regenerate the score write-back over the whole corpus: `make score-cases`.

---

## 4. Evaluation

Test split, 3 seeds, mean ± std. Paired bootstrap, 2,000 resamples, 95% CI.

### 4.1 Headline

| Metric | Value |
|---|---|
| **Test AUC-PR** | **0.8720 ± 0.0136** |
| Test AUC-ROC | 0.9541 ± 0.0060 |
| Realistic-imbalance stream AUC-PR | 0.8944 ± 0.0055 |
| Typology macro-F1 (structural) | 0.337 ± 0.015 (chance 0.142) |
| Test AUC-PR range across seeds | 0.8595 – 0.8865 |

### 4.2 Against the other arms

| Arm | Params | Test AUC-PR |
|---|---:|---|
| Graph transformer (virtual node) | 1,021,530 | **0.8877 ± 0.0190** |
| GATv2 + weighted BCE (ablation) | 628,058 | 0.8775 ± 0.0106 |
| GIN / GINE | 1,021,277 | **0.8801 ± 0.0056** |
| **GATv2 (primary)** | **628,058** | **0.8720 ± 0.0136** |
| GATv2 without positional encodings | 628,058 | 0.8696 ± 0.0049 |
| **MLP control — no message passing** | 1,008,218 | **0.8017 ± 0.0407** |
| GraphSAGE | 613,466 | 0.7861 ± 0.0101 |
| GATv2 without edge features | 628,058 | 0.7582 ± 0.0236 |
| GCN | 416,858 | 0.7137 ± 0.0213 |

**Two arms out-score the primary and it stays primary.** Neither margin is significant on
the paired difference, and D-064 records the evidence and exactly what would change the
choice. Reporting GATv2 as best would be a misreading of this table.

### 4.3 The gate, and the number that bounds every downstream claim

**GATv2 beats the MLP control by +0.070 AUC-PR, excluding zero at all three seeds. The
gate passed.**

**But the MLP control reaches 0.8017 with no message passing at all.** It is a DeepSets
model over case-local node features. So:

> **Every claim about what graph structure contributes — in this project, in the fusion
> layer, in the paper — is a claim about 0.07, not about 0.87.**

At two epochs the gap read 0.16, which is what an under-tuned control would have reported.
The control was trained to convergence specifically so that this number would be honest.

### 4.4 Attention alignment and the Phase 8 forecast

A **linear probe on the pooled tokens the fusion layer consumes reaches 0.33 structural
macro-F1**. `fan_out`, `gather_scatter` and `cycle` are recoverable from those tokens;
**`stack` and `random` are not.** This is the forecast for what a fusion layer could
possibly transmit, and it is a caution rather than a promise.

Attention-alignment figures (mass on laundering-path accounts, per-typology lift) are in
`artifacts/metrics/encoder/encoder_report.json` and PHASE_LOG Phase 7.

### 4.5 Ablations — including two nulls, kept and reported

| Ablation | Effect on test AUC-PR | Significant? | Decision |
|---|---:|---|---|
| Edge features removed | **−0.114** | **yes** | edge features kept |
| Positional encodings removed | −0.002 | **no** | retained anyway (D-066) |
| Focal loss → weighted BCE | **+0.006** | **no** | focal kept as default (D-065) |

Positional encodings cost parameters and buy nothing measurable. Weighted BCE very
slightly *out*-scores focal loss. Both nulls are reported rather than dropped
(invariant 7).

---

## 5. Limitations

- **One substrate, and it is synthetic.** AMLworld is agent-based simulator output.
  Nothing here is evidence of performance on real transaction data. Elliptic2 has never
  been ingested.
- **One dataset variant.** HI-Small only. The other five AMLworld variants load but have
  no pinned figures.
- **A case is a fragment.** The 48-hour extraction window keeps ~65% of a laundering
  stream's transactions on average (D-019). `typology` means "part of a stream of this
  typology", not "exhibits it in full", and the encoder is scored against that label.
- **Only 17.7 days of data**, so the temporal split has very little room and the
  train/test gap is short by any real deployment standard.
- **Typology recovery is weak in absolute terms** (0.337 macro-F1 against 0.142 chance),
  and it is weak *unevenly* — see §4.4.
- **A 9-point Fact Coverage spread by typology** exists downstream in the corpus
  (`RESULTS.md` §4.3); the encoder does not correct it.
- **Between-run variance on the development host is large.** The host is a laptop that
  throttles. Encoder numbers come from a full sweep, but the Phase 13 latency figures on
  the same machine showed a 2× between-run spread.
- **Attention weights are not explanations.** They are reported as a measured alignment
  statistic, not as an account of why a case scored as it did.

---

## 6. Misuse

**In addition to every prohibition in `docs/ETHICS.md` §2:**

**This model must not be used to score, rank, flag or triage real accounts or real
customers.** It is trained on synthetic data from one simulator, on one 17.7-day window,
with a case-level label that means "part of a stream of this typology". A score it emits
about a real account is not evidence of anything, and the model provides no calibration
that would let anyone find that out from its output.

**Its score must not be presented as a risk assessment of a person.** `gnn_risk_score` is
a case-subgraph statistic. Attaching it to an individual and treating it as a
characteristic of that individual is a category error with consequences for a real person.

**It must not be used as an autonomous filter.** The +0.070 margin over a control with no
message passing at all is the honest measure of what the graph structure buys here. A
deployment that treats 0.87 AUC-PR as licence to drop the low-scoring tail is acting on a
number that was measured on synthetic data and never validated on real data.

**Its outputs must not be described as explainable.** The typology head recovers three of
eight typologies above chance and fails on `stack` and `random`. Presenting a typology
prediction to an investigator without that breakdown overstates what the model knows.

---

## 7. Reproduction

```bash
make encoder-features   # build the feature cache from the frozen splits
make train-encoder      # 9 arms x 3 seeds, ~6.5 h on one 4 GB card
make encoder-gate       # the tests
uv run python scripts/07c_report_tables.py   # render the report as tables
```

Expected: test AUC-PR 0.8720 ± 0.0136 for `gatv2`, gate PASSED against the MLP control.
GPU non-determinism means the digits will not match exactly; the tolerance policy is in
`docs/REPRODUCTION.md` § Tolerances. **The gate outcome must reproduce; the fourth decimal
place need not.**

---

## 8. Citation

See `CITATION.cff`. Cite the substrate as well — Altman et al., NeurIPS Datasets &
Benchmarks 2023.
