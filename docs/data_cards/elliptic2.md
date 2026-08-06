# Data card — Elliptic2

**Substrate role:** real-world demonstration substrate.
**Card written:** 2026-08-01 (Phase 1) · **Loader:** `src/g2t_aml/data/loaders/elliptic2.py`
**Status: ACCESS NOT YET GRANTED.** Everything below marked *(documented)* comes from the
paper and the official tooling, not from files we hold. Nothing here has been verified
against data. Observed statistics are absent by necessity, and the loader's tests skip.

---

## 1. Source and citation

| | |
|---|---|
| Name | Elliptic2 |
| Authors | Bellei, Fenton, et al. (Elliptic, MIT-IBM Watson AI Lab) |
| Publication | *The Shape of Money Laundering: Subgraph Representation Learning on the Blockchain with the Elliptic2 Dataset*, KDD Workshop on Machine Learning in Finance, 2024 ([arXiv:2404.19109](https://arxiv.org/abs/2404.19109)) |
| Access request | <https://www.elliptic.co/elliptic2> |
| Official tooling | <https://github.com/MITIBMxGraph/Elliptic2> |
| Nature | **Real Bitcoin blockchain data**, clustered and anonymised |

```bibtex
@inproceedings{bellei2024shape,
  title     = {The Shape of Money Laundering: Subgraph Representation Learning on the
               Blockchain with the Elliptic2 Dataset},
  author    = {Bellei, Claudio and Fenton, Muhua Xu and others},
  booktitle = {KDD Workshop on Machine Learning in Finance},
  year      = {2024}
}
```

---

## 2. Licence and redistribution — UNRESOLVED, treat as closed

**We could not locate a licence for the Elliptic2 *data* (checked 2026-08-01).**

- The official tooling repository `MITIBMxGraph/Elliptic2` is **Apache-2.0**, but that
  covers the *code*; the repository states nothing about the dataset's terms.
- The dataset itself sits behind a request form at `elliptic.co/elliptic2`. The terms
  presumably arrive with the access grant.
- A Kaggle mirror exists (`ellipticco/elliptic2-data-set`) whose licence field we could
  not read programmatically.

**Position taken:** `redistributable=False` in the registry, and the substrate is treated
as **not redistributable** until we hold a written statement to the contrary.

**Action required before Phase 14.** When access is granted, read the agreement actually
presented and update this section, the registry `licence` and `redistributable` fields,
and `PHASE_LOG.md`. Discovering a restriction at release time is exactly the expensive
failure this card exists to prevent. Note also that redistribution is a *separate* question
from whether derived artifacts may be released: even a fully closed dataset usually permits
publishing metrics, but that must be confirmed, not assumed.

Until then, Phase 14 must assume it can ship **no** Elliptic2 rows, **no** interim Parquet
derived from them, and possibly no per-subgraph outputs.

---

## 3. Files *(documented)*

Once unzipped into `data/raw/elliptic2/`:

```
background_edges.csv      ~196M edges
background_nodes.csv      ~49M cluster nodes with anonymised features
connected_components.csv  subgraph membership and licit/suspicious labels
edges.csv                 edges of the labelled subgraphs
nodes.csv                 nodes of the labelled subgraphs
```

No checksums are pinned — we have never seen the files. `verify()` reports `MISSING` and
`scripts/01_ingest.py` **skips cleanly with exit 0** rather than failing, because Phase 1
does not block on an access-gated substrate. Checksums must be pinned on first ingest.

### Column names are probed, never guessed

The official tooling has used more than one spelling across releases, so the loader probes
a candidate list per role (`clusterId`/`cluster_id`/`nodeId`/…, `ccId`/`componentId`/…,
`ccLabel`/`label`/…) and raises `Elliptic2SchemaError` — *"Refusing to guess by position"* —
if none matches. Positional guessing would silently mislabel an entire graph.

---

## 4. What the data is *(documented)*

- **122K labelled subgraphs** within a background graph of ~49M nodes and ~196M edges.
- **The unit is a *cluster*** — a set of Bitcoin addresses believed to share ownership.
  **Not an address, not a wallet, not a transaction, not an account.** The loader sets
  `node_type = "cluster"` and the Phase 4/5 templates must use language that respects
  this. "This account received…" is factually wrong for this substrate.
- **Labels are subgraph-level**, `licit` or `suspicious`. Per the construction procedure: a
  subgraph is *licit* if its senders and receivers are all licit; *suspicious* if its
  receivers are licit but its senders are illicit.

**A caution the narrative layer must respect:** `suspicious` is a property of the money's
**provenance**, not a proven offence by any cluster in the subgraph. It licenses "received
funds originating from clusters associated with illicit activity", not "this entity
laundered money". The loader normalises both string and 1/2 integer label encodings, and
**rejects** any unrecognised value rather than defaulting it — defaulting to `licit` is how
a suspicious subgraph would become invisible.

---

## 5. Availability mask — almost everything is masked off

Invariant 4. Elliptic2 is the reason invariant 4 exists.

| Field | Value | Justification |
|---|---|---|
| `absolute_timestamps` | **false** | No wall-clock times. Temporal information is coarse step indices at best |
| `fine_temporal_resolution` | **false** | Nothing at hour granularity or better. **No narrative may say "within 22 hours"** for this substrate |
| `monetary_amounts` | **false** | No amounts of any kind |
| `currencies` | **false** | Bitcoin throughout; there is no currency *field*, and there are no fiat conversions |
| `institution_identity` | **false** | No banks. Blockchain clusters have no institutional affiliation |
| `entity_types` | **false** | **No "mixer", no "exchange", no "merchant".** These labels do not exist in the data. Inventing one is the most tempting and most damaging hallucination available on this substrate, because it sounds exactly like what an analyst would write |
| `node_labels` | **true** | The subgraph-level licit/suspicious label. The one thing supplied |
| `typology_ground_truth` | **false** | No structural typology labels. `CanonicalGraph.typology` is **None**, which is a different claim from AMLworld's `"unclassified"`: None means no typology truth exists at all; `unclassified` means it exists and reports no match |
| `semantic_node_features` | **false** | **Features are anonymised.** Column meanings are not published |

### On the anonymised features

Node feature columns are carried verbatim, with their opaque source names, and are never
renamed or interpreted. **No column index may ever be mapped to a named financial
quantity** — not in code, not in a template, not in a comment. The features still reach
the GAT encoder as numbers; what is forbidden is attaching meaning to them in text.

The Phase 0 config recorded `node_features: true` on the reasoning that feature columns
exist. That was corrected in Phase 1 to `false`: the mask governs what may be *asserted*,
and an anonymised column licenses no assertion at all.

---

## 6. Observed versus published statistics

| Quantity | Published | Observed |
|---|---:|---|
| Labelled subgraphs | ~122,000 | *not yet measured* |
| Background nodes | ~49,000,000 | *not yet measured* |
| Background edges | ~196,000,000 | *not yet measured* |

The published figures are rounded, so `test_elliptic2_has_the_published_number_of_labelled_subgraphs`
asserts a tolerance rather than equality. Fill this table on first successful ingest.

---

## 7. Known limitations

1. **Access-gated**, and possibly not redistributable. See §2.
2. **Anonymised features** — the central limitation. Most of what makes a SAR narrative
   useful (amounts, timing, counterparty types) simply is not derivable.
3. **Clusters, not entities.** Cluster membership is *inferred* by heuristics. A cluster
   may merge addresses of different real owners, or split one owner across clusters. Any
   claim about "an entity" inherits that uncertainty.
4. **Subgraph-level labels only.** No node is individually labelled, so per-node claims
   are unsupported.
5. **`suspicious` ≠ criminal.** See §4.
6. **Scale.** The background graph does not fit in 32 GB alongside anything else. Access
   is lazy: `load_background_graph()` returns Polars `LazyFrame` scans and `build_subgraph`
   pushes its filter into the scan. `component_statistics` refuses to run above
   `max_nodes` rather than exhausting memory.
7. **No negative-class guarantee.** A `licit` subgraph is one not identified as
   suspicious, which is not the same as one verified clean.
8. **This substrate cannot support the project's richest narratives**, by construction.
   That is the point of carrying it: it demonstrates the fact layer correctly *withholding*
   claims, and the availability mask doing real work rather than being decoration.

---

## 8. Our preprocessing *(planned; unexecuted)*

`scripts/01_ingest.py data=elliptic2` will:

1. Verify the five files (checksums to be pinned on first ingest).
2. Load `connected_components.csv` into `(subgraph_id, node_id, label)`, normalising and
   validating labels.
3. Open lazy scans over the background tables.
4. Build a representative canonical subgraph to prove the path end to end.
5. Compute statistics, write Parquet, manifest and statistics report.

Materialising all 122K subgraphs is **Phase 2** (case extraction), not Phase 1.

At present the run logs the access instructions and exits 0 with
`ingest_skipped.json` recording the reason.

---

## 9. Access request status

| | |
|---|---|
| Requested on | **not yet requested as of 2026-08-01** |
| Granted on | — |
| Terms received | — |

Record the request date here and in `PHASE_LOG.md` when it is made. Phase 1's gate does
not depend on it; Phase 12's ablation matrix across both substrates does.
