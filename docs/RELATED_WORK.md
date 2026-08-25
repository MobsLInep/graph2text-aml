# Related work, prior art, and where the novelty actually is

**Written 2026-08-25, after a systematic paper and patent search.** This document exists
because a contribution claim that has not been checked against the literature is a claim
about our reading habits rather than about the field.

**It changed what we think our contribution is.** Read §5 before writing the paper's
introduction.

---

## 0. Summary

| Question | Answer |
|---|---|
| Does a public paired *(graph, SAR narrative)* corpus exist? | **No.** The premise holds. |
| Has anyone done graph-conditioned SAR narrative generation? | **Yes — SCAFDS (2026), at institution level.** §1.1 |
| Is the fusion layer architecturally novel? | **No.** It is the GraphToken/LLaGA recipe. §4 |
| Has anyone measured SAR narrative faithfulness? | **Partially.** SCAFDS reports 60.0% assertion traceability. Nobody reports a per-narrative binary. §1.1, §5 |
| Has anyone measured exculpatory omission? | **No.** §5 |
| Are there blocking patents? | **Three close ones, two held by Actimize.** §3 |

---

## 1. Closest prior work

### 1.1 SCAFDS — the closest thing that exists

> Uddin, M. N. *SCAFDS: Edge-Feature Graph Attention for Interbank Fraud Detection with
> Attribution-Grounded SAR Generation.* arXiv:2605.18913 (2026).
> USPTO Provisional Patent Application No. 64/061,083, filed 2026-05-08.

**This paper independently arrives at four of our design decisions.** It must be cited and
distinguished explicitly; a reviewer who knows it and does not see it discussed will assume
we do not.

| SCAFDS | Graph2Text AML |
|---|---|
| Edge-feature-informed graph attention | GATv2 with edge features |
| Edge-feature ablation: **+0.266 AUPRC** | Ours: **+0.114 AUC-PR** — same qualitative finding |
| Randomised-edge control ⇒ 0.248, indistinguishable from no-edge (p=0.91, Wilcoxon) | A1 derangement control (D-071) — same logic, unrun |
| Attribution-conditioned SAR generation with per-assertion grounding | Fact-record-grounded generation with a three-valued checker |
| **60.0% of assertions traceable** to a numerical pipeline output | Zero-Hallucination Rate, per-narrative binary |
| FinCEN Form 111 formatted output | `docs/annotation/` protocol, four-part structure |

**Their Track B results** (institution-level, synthetic FDIC-aligned interbank graph, 8,103
institutions, 169,800 directed edges, **15% positive rate**):

| Model | AUPRC | AUROC | F1 |
|---|---|---|---|
| GCN | 0.240 ± 0.016 | 0.576 | 0.271 |
| GAT (node-only) | 0.260 ± 0.017 | 0.594 | 0.279 |
| LineMVGNN | 0.298 ± 0.018 | 0.635 | 0.323 |
| GraphSAGE-AML (from LaundroGraph) | 0.356 ± 0.020 | 0.665 | 0.360 |
| **SCAFDS** | **0.515 ± 0.032** | **0.802** | **0.508** |
| SCAFDS-NoEdge (ablation) | 0.249 ± 0.016 | 0.585 | — |

**Three things separate us, and only three.**

1. **Level of analysis.** SCAFDS models *interbank institution* fraud propagation, with
   edges carrying a fraud co-occurrence frequency derived from the FinCEN SAR registry. We
   model *account-level* transaction subgraphs. Different graph, different unit of report.
2. **The grounding record.** Theirs is a three-layer SHAP attribution record (transaction
   SHAP, network SHAP, temporal attention). Ours is an extracted `case_facts` record with a
   frozen schema and a three-valued checker. **A SHAP value says a feature mattered; it does
   not say a stated fact is true.** That is the distinction our contribution rests on.
3. **The conditioning mechanism.** SCAFDS constrains the LLM by **threshold suppression** —
   assertions whose grounding attribution falls below tau are dropped from the prompt. We
   project graph tokens into the LM's embedding space. Prompt-level suppression versus
   representation-level conditioning.

**Do not claim we are the first to ground a SAR narrative in a detector's own output.
SCAFDS is.**

### 1.2 The LLM baselines, already published on our substrate

> *Exploring the In-Context Learning Capabilities of LLMs for Money Laundering Detection in
> Financial Graphs.* The Vanguard Group. arXiv:2507.14785 (2025).

**This is our B3/B4 baseline, already run on the IBM AMLworld substrate with the same eight
typologies.** Subgraphs are extracted around transactions, serialised to tagged text, and
passed to GPT-4o with few-shot exemplars. 2,000 balanced examples.

| | Accuracy | Precision | Recall |
|---|---|---|---|
| GPT-4o, few-shot | 63.7 ± 3.3 | 62.7 ± 4.4 | 67.7 ± 4.4 |

Per-typology precision / recall (%):

| Typology | Precision | Recall |
|---|---:|---:|
| fan_in | 73.8 | 81.5 |
| fan_out | 75.1 | 80.3 |
| gather_scatter | 66.2 | 70.7 |
| scatter_gather | 64.9 | 68.8 |
| cycle | 62.5 | 65.9 |
| bipartite | 55.0 | 56.1 |
| **stack** | **52.0** | 58.0 |
| **random** | **51.5** | 53.5 |

**This independently reproduces our Phase 7 finding.** Our linear probe on pooled encoder
tokens found `fan_out`, `gather_scatter` and `cycle` recoverable and **`stack` and `random`
not**. Their prompted LLM bottoms out on exactly `random` (51.5) and `stack` (52.0).

Two different methods, two different representations, the same typologies at the bottom.
That is convergent evidence that **the difficulty is in the typologies, not in our encoder**
— which is a stronger claim than either paper can make alone, and it should be stated.

They explicitly **do not evaluate explanation faithfulness**.

### 1.3 Other narrative-generation work

| Work | What it does | Faithfulness reported |
|---|---|---|
| *Co-Investigator AI*, arXiv:2509.08380 (2025) | Agentic SAR narrative drafting, expert-rated | **None.** Narrative completeness 3.52/5 (70%); 87% on specific typologies; efficiency 3.06/5 (61%). Cites literature hallucination "frequently exceeding 20–30%" but reports no rate of its own. |
| *Regulatory Graphs and GenAI*, arXiv:2506.01093 (2025) | GNN + LLM → NL compliance explanations, Elliptic | Discusses auditability; **no quantified hallucination rate** |

---

## 2. Detection benchmarks — what the numbers are, and why ours do not compare

### 2.1 AMLworld HI-Small — our substrate, but a different task

Minority-class F1 (%), **transaction/edge classification**:

| Model | HI-Small F1 | Source |
|---|---:|---|
| GIN | 28.70 ± 1.13 | Altman et al. 2023 |
| GIN+EgoIDs | 39.65 ± 4.73 | Egressy et al. 2024 |
| R-GCN | 41.78 ± 0.48 | Egressy et al. 2024 |
| GIN+ReverseMP | 46.79 ± 4.97 | Egressy et al. 2024 |
| GIN+EU | 47.73 ± 7.86 | Altman et al. 2023 |
| GIN+Ports | 54.85 ± 0.89 | Egressy et al. 2024 |
| PNA | 56.77 ± 2.41 | Altman et al. 2023 |
| Multi-GIN | 57.15 ± 4.99 | Egressy et al. 2024 |
| GFP + LightGBM | 62.86 ± 0.25 | Altman et al. 2023 |
| GFP + XGBoost | 63.23 ± 0.17 | Altman et al. 2023 |
| Multi-PNA | 64.59 ± 3.60 | Egressy et al. 2024 |
| Multi-GIN+EU | 64.79 ± 1.22 | Egressy et al. 2024 |
| **Multi-PNA+EU** | **68.16 ± 2.65** | Egressy et al. 2024 — **published maximum** |

Supporting detail (Altman et al., HI-Small): PNA precision 58.48 / recall 53.15; GIN+EU
42.29 / 55.41; GIN 27.40 / 38.16.

**Why our 0.8720 AUC-PR is not "better than 68.16".** Their unit is a transaction and their
metric is minority-class F1 at a threshold. Our unit is a 2-hop case subgraph and our metric
is AUC-PR, threshold-free. Different task, different unit, different metric. **Any table
placing the two side by side is wrong.**

### 2.2 Elliptic2 — the metric trap

> Bellei et al., *The Shape of Money Laundering.* KDD MLF 2024. Subgraph classification.

| Method | Test F1 (micro) | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|---:|
| GNN-Seg | 0.398 | 0.026 | 0.537 |
| Sub2Vec | **0.944** | **0.022** | 0.496 |
| GLASS | 0.933 | **0.208** | 0.889 |

**Read that Sub2Vec row carefully.** It posts the highest micro-F1 in the table while its
PR-AUC (0.022) and ROC-AUC (0.496) say it is indistinguishable from random. Micro-averaged
F1 under extreme imbalance is close to accuracy and is nearly uninformative here.

Any secondary source quoting *"GLASS achieves ~93% on Elliptic2"* is quoting that
misleading column. **The meaningful figure is PR-AUC 0.208.** GLASS's confusion matrix
(TP 245, FP 765, FN 46, TN 11,125) gives precision 24.3% at recall 84.2%.

**And our 0.8720 must not be set against 0.208.** Ours is stratified synthetic data at
14.7% positive prevalence; theirs is real Bitcoin at natural prevalence. Even our
realistic-imbalance stream (7.3%, AUC-PR 0.8944) is synthetic. This project never obtained
Elliptic2 (D-099), so we have no number on their substrate at all.

### 2.3 Graph-to-text faithfulness — the one structurally valid comparison

> He et al., *G-Retriever.* NeurIPS 2024. GraphQA benchmark.

| | Node validity | Edge validity | **Fully valid output** |
|---|---:|---:|---:|
| LLM + graph prompt tuning | 31% | 12% | **8%** |
| G-Retriever | 77% | 76% | **62%** |

**"Fully valid" is a per-output, all-or-nothing binary — structurally identical in shape to
our Zero-Hallucination Rate.** This is the only external number our headline metric can be
placed beside.

**We cannot use it yet.** Our 1.0000 is Bronze, a deterministic template that renders from
the fact record with every formatter shipping its inverse (D-040). It is faithful *by
construction*. Comparing a template to a learned model would be the single most misleading
claim available to this project. **The comparison becomes legitimate the day a trained arm
is scored — that is Gate 8, and it is open.**

---

## 3. Patent landscape

Three patents are close enough to matter. **Two are held by Actimize (NICE Actimize), a
major AML vendor**, and one of ours-adjacent has a provisional already filed.

| Patent | Assignee | Date | What is claimed |
|---|---|---|---|
| **US12469007B2** — *Automatic generation of a two-part readable SAR from high-dimensional data in tabular form* | **Actimize Ltd** (Butvinik, Pati) | priority **2020-08-06** | Per-transaction-type NLG models; hierarchical attention over keywords/sentences to summarise; copy mechanism for OOV words. **Tabular input, no graph, no LLM.** |
| **US12536544B2** — *Automated validation of SAR narratives using generative AI* | **Actimize Ltd** (Bathula) | filed **2024-01-25** | Parse a **JSON object** of the narrative's entities → load a prompt template → **compare the SAR narrative to the SAR data** via calls to a generative-AI service → store the validation. |
| **US11176557B2** — *Neural network host platform for generating automated SARs* | Bank of America | — | CNN over compressed transaction history → suspicious-event **probability score**. Not narrative generation. |

Also noted: **US11522881**, *Structural graph neural networks for suspicious event
detection* (anomalous edge detection in dynamic graphs).

**US12536544B2 is the one to read closely.** Its claim 1 is, in substance, our **Method B
LLM claim extractor** and our **Silver verification loop**: validating a SAR narrative
against structured record data using an LLM under a prompt template. It covers validation
only — it neither generates the narrative nor touches a graph — but anyone reading our
Phase 5 and Phase 10 design will see the resemblance.

**This is a note for the authors, not legal advice.** If any part of this work is ever
commercialised, these three should go to counsel first. For publication, they are prior art
to cite.

---

## 4. The fusion layer is not novel, and the brief was wrong to say it was

`CLAUDE.md` §1 states: *"The fusion layer is the technical novelty: how graph structure gets
injected into the language model's embedding space."*

**That claim will not survive review.** Encoding a graph with a GNN, projecting it through a
learned adapter, and prepending the result as soft tokens to an LLM is an established
recipe with at least five published instances predating this work:

| Work | Mechanism |
|---|---|
| **GraphToken** — *Let Your Graph Do the Talking*, arXiv:2402.05862 (Google) | GNN encoder → fixed number of token embeddings → prepended to the LLM's input sequence |
| **LLaGA** — arXiv:2402.08170 | Structure-aware token sequences → learned projector → LLM token embedding space |
| **GraphGPT** | Contrastive text-grounded GNN → MLP projector → graph tokens |
| **G-Retriever** — arXiv:2402.07630 (NeurIPS 2024) | Retrieval + graph encoder → soft prompt |
| **Graph2Token** | Molecule graphs → LLM token space |

Our `PrefixFusion` with three projector variants is a competent instance of this family. It
is **not** the contribution. Presenting it as one invites a reviewer to produce GraphToken
and reject on novelty.

**What is defensible:** that nobody has applied it to SAR narrative generation, and — more
importantly — that we can *measure whether it works*, via the A1 derangement control
(D-071). SCAFDS's threshold-suppression approach cannot answer "does the model read the
graph token", because it has no graph token. **Gate 8 is a question no prior work in this
list has posed.**

---

## 5. Where the novelty actually is

After the search, four claims survive. All four are about **measurement and data**, not
architecture.

**1. There is no public paired (graph, SAR narrative) corpus, and we built one.**
Verified against AMLworld, Elliptic2, TransXion, AMLGentex and the Vanguard work — all are
*detection* datasets with no narratives. Real SARs are confidential by statute. The
three-tier Bronze/Silver/Gold construction, with one harness gating all three identically
(D-037), has no precedent we could find.

**2. The fact layer is a bidirectional measurement instrument.** The same code that
generates the corpus scores faithfulness at evaluation time, run in reverse, and it is
calibrated against an **independent oracle** because the round trip alone cannot catch an
extractor bug — three injected bugs left it at 100% SUPPORTED (D-034). No related work here
validates its own faithfulness checker against anything external.

**3. Zero-Hallucination Rate as a per-narrative binary, plus a nine-class taxonomy with a
separately-reported Critical Error Rate.** SCAFDS reports 60.0% *assertion* traceability —
an average over claims. One fabricated fact makes a SAR unfileable regardless of the rest,
which is why the per-narrative binary is the right unit (D-077). Nobody else reports one.

**4. H9 — omission of an exculpatory fact — is measured by nobody else.**
SCAFDS measures whether what *is* said is traceable. We measure what is wrongly *left out*,
and found our own template corpus omits it in **91.79%** of narratives. It is the failure
mode least likely to survive human review, because a reviewer cannot notice the absence of
something they were never shown (`docs/ETHICS.md` §3.3). This is the most defensible
individual finding the project currently holds, and it is a negative result about our own
artifact.

### Recommended reframing

> **From:** "the fusion layer is the technical novelty"
> **To:** "the corpus and the measurement instrument are the contribution; the fusion layer
> is the system under test, and Gate 8 is the experiment that tests it."

That framing is honest, is supported by what is already built and verified, and does not
depend on a training run that has not happened.

---

## 6. Bibliography

- Altman, E. et al. *Realistic Synthetic Financial Transactions for Anti-Money Laundering
  Models.* NeurIPS Datasets & Benchmarks, 2023. arXiv:2306.16424
- Egressy, B. et al. *Provably Powerful Graph Neural Networks for Directed Multigraphs.*
  AAAI 2024. arXiv:2306.11586
- Bellei, C. et al. *The Shape of Money Laundering: Subgraph Representation Learning on the
  Blockchain with the Elliptic2 Dataset.* KDD MLF 2024. arXiv:2404.19109
- He, X. et al. *G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding
  and Question Answering.* NeurIPS 2024. arXiv:2402.07630
- Perozzi, B. et al. *Let Your Graph Do the Talking: Encoding Structured Data for LLMs*
  (GraphToken). arXiv:2402.05862
- Chen, R. et al. *LLaGA: Large Language and Graph Assistant.* arXiv:2402.08170
- Uddin, M. N. *SCAFDS: Edge-Feature Graph Attention for Interbank Fraud Detection with
  Attribution-Grounded SAR Generation.* arXiv:2605.18913, 2026
- The Vanguard Group. *Exploring the In-Context Learning Capabilities of LLMs for Money
  Laundering Detection in Financial Graphs.* arXiv:2507.14785, 2025
- *Co-Investigator AI: Agentic AI for AML Compliance Narratives.* arXiv:2509.08380, 2025
- *Regulatory Graphs and GenAI for Real-Time Transaction Monitoring and Compliance
  Explanation in Banking.* arXiv:2506.01093, 2025
- US12469007B2, Actimize Ltd — automatic two-part SAR generation from tabular data
- US12536544B2, Actimize Ltd — automated SAR narrative validation using generative AI
- US11176557B2, Bank of America — neural network platform for automated SARs
- US11522881 — structural GNNs for suspicious event detection

---

## 7. How this document was produced

Systematic search over arXiv, NeurIPS/AAAI/KDD proceedings, Google Patents and USPTO
full-text, 2026-08-25. Every number in §1 and §2 was extracted from the primary PDF, not
from an abstract or a secondary summary — §2.2 exists because a secondary summary of the
Elliptic2 result was wrong in a way that would have flattered us.

**Gaps.** The search was English-language and did not cover EPO/WIPO filings, vendor white
papers, or any non-public FinCEN/FCA material. Absence of evidence here is weaker than the
§5 claims imply, and a formal freedom-to-operate opinion is not something this document
constitutes.
