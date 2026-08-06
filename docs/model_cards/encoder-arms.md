# Model card — encoder comparison and ablation arms

**Eight non-primary encoder checkpoint families**, released so that the comparison in the
paper is auditable rather than merely reported. The primary arm has its own card:
[`gatv2-case-encoder.md`](gatv2-case-encoder.md).

| | |
|---|---|
| Card written | 2026-08-06 (Phase 14), for release `v0.1.0` |
| Checkpoints | 8 arms × 3 seeds = 24, plus the 3 primary = 27 total, ~78 MB |
| Licence | Apache-2.0 (Results under CDLA-Sharing-1.0 §3.5) |
| Code | `src/g2t_aml/models/encoder/arms.py` |

---

## 1. Why these are released at all

Two of these arms out-score the primary and the primary stays primary (D-064). One of them
— the MLP control — is the number that bounds every claim the project makes about graph
structure. A reader who cannot check those claims against the actual checkpoints has to
take the table on trust.

**Every arm overrides `message_passing` and nothing else.** Same feature space, same
training loop, same seeds, same loss, same early-stopping criterion. That constraint is what
makes §3 a comparison of message passing rather than of eight differently-tuned models, and
it is enforced by `BaseEncoder` rather than by discipline.

---

## 2. The arms

### Comparison arms — different message passing

| Arm | What it is | Params |
|---|---|---:|
| `gin` | GINE, edge-conditioned GIN | 1,021,277 |
| `sage` | GraphSAGE, mean aggregation | 613,466 |
| `gcn` | Graph convolutional network | 416,858 |
| `graph_transformer` | Virtual-node graph transformer | 1,021,530 |
| `mlp` | **The control.** DeepSets over case-local node features. **No message passing at all.** | 1,008,218 |

### Ablation arms — GATv2 with one thing removed

| Arm | What changed |
|---|---|
| `gatv2_no_pe` | Laplacian and random-walk positional encodings removed |
| `gatv2_no_edge` | Edge features removed |
| `gatv2_bce` | Focal loss replaced with class-weighted BCE |

---

## 3. Evaluation

Test split, 3 seeds (42/43/44), mean ± std, paired bootstrap 2,000 resamples.

| Arm | Test AUC-PR | Test AUC-ROC | Realistic stream | Typology macro-F1 |
|---|---|---|---|---|
| `graph_transformer` | **0.8877 ± 0.0190** | 0.9648 | 0.8973 | — |
| `gin` | **0.8801 ± 0.0056** | 0.9585 | 0.8409 | — |
| `gatv2_bce` | 0.8775 ± 0.0106 | 0.9582 | 0.8995 | — |
| *`gatv2` (primary)* | *0.8720 ± 0.0136* | *0.9541* | *0.8944* | *0.337 ± 0.015* |
| `gatv2_no_pe` | 0.8696 ± 0.0049 | 0.9552 | 0.8865 | — |
| **`mlp` (control)** | **0.8017 ± 0.0407** | 0.9211 | 0.7992 | — |
| `sage` | 0.7861 ± 0.0101 | 0.9066 | 0.8617 | **0.356 ± 0.022** |
| `gatv2_no_edge` | 0.7582 ± 0.0236 | 0.8985 | 0.8630 | — |
| `gcn` | 0.7137 ± 0.0213 | 0.8862 | 0.7786 | 0.319 ± 0.024 |

Chance for typology macro-F1 is 0.142.

### Paired differences from the primary

| `gatv2` minus | Mean difference | Excludes zero at every seed |
|---|---:|---|
| `mlp` (the gate) | **+0.070** | **yes — gate PASSED** |
| `sage` | +0.086 | yes |
| `gcn` | +0.158 | yes |
| `gatv2_no_edge` | **+0.114** | yes |
| `gatv2_no_pe` | +0.002 | **no — null result** |
| `gatv2_bce` | −0.006 | **no — null result** |
| `gin` | −0.008 | no |
| `graph_transformer` | −0.016 | no |

---

## 4. What these arms establish, and what they do not

**The MLP control is the important row.** It reaches 0.8017 with no message passing at
all — a DeepSets model over node-local summary statistics gets within 0.07 of the primary.
Read every downstream claim about topology against 0.80, never against zero. At two epochs
the gap read 0.16; the control was trained to convergence so that this number would not
flatter the method.

**Two arms beat the primary and neither significantly.** `graph_transformer` (+0.016) and
`gin` (+0.008) out-score GATv2 with overlapping intervals. GATv2 remains primary on the
grounds recorded in D-064; the alternative reading — that the choice of message passing
barely matters on this task — is also consistent with this table and is stated in the
paper.

**Two nulls are kept and reported** (invariant 7): positional encodings buy +0.002 (not
significant) and are retained anyway (D-066); focal loss is 0.006 *worse* than weighted BCE
(not significant) and is kept as the default (D-065). Neither was dropped for being
inconvenient.

**Edge features are the one clearly load-bearing design choice**: removing them costs
0.114, the largest effect in the table.

**`sage` posts the best typology macro-F1 (0.356) while ranking seventh on AUC-PR.** The
two heads are not measuring the same thing and the arm ordering differs between them.

---

## 5. Limitations

Everything in [`gatv2-case-encoder.md`](gatv2-case-encoder.md) §5 applies unchanged to all
eight arms: one synthetic substrate, one variant, a 17.7-day window, case labels that mean
"part of a stream of this typology", and a laptop-class host.

Additionally:

- **`mlp` is a control, not a proposal.** It is released so its number can be checked, not
  as a recommended model.
- **The ablation arms are not tuned for their configuration.** `gatv2_no_edge` uses the
  primary's hyperparameters with edge features removed. A model designed without edge
  features from the start might do better, and this table does not bound that.
- **Three seeds is three seeds.** `gatv2_no_pe` has a std of 0.0049 and `mlp` has 0.0407 —
  an eight-fold difference in spread across arms that share a training loop. Small margins
  between adjacent rows should not be read as orderings.

---

## 6. Misuse

**All prohibitions in `docs/ETHICS.md` §2 and in
[`gatv2-case-encoder.md`](gatv2-case-encoder.md) §6 apply to every arm here.** In
particular, none of these checkpoints may be used to score, rank or triage real accounts,
and none of their scores may be attached to an individual as a risk assessment.

**One prohibition specific to this card: do not report a number from these checkpoints
without its control.** The comparison arms exist to bound a claim. Quoting
`graph_transformer` at 0.8877 as a headline, without the MLP control at 0.8017 beside it,
converts a bounded finding into an unbounded one. That is the specific misuse this card is
written to prevent.

---

## 7. Reproduction

```bash
make train-encoder                              # all 9 arms x 3 seeds, ~6.5 h on one 4 GB card
uv run python scripts/07c_report_tables.py      # renders every table above from the report
```

The tables in this card are rendered from `artifacts/metrics/encoder/encoder_report.json`
by that script rather than retyped. Tolerances: `docs/REPRODUCTION.md` § Tolerances.
