# Data cards

One card per substrate, written in Phase 1 alongside the ingestion code:

- `amlworld.md` — IBM AMLworld (Altman et al., NeurIPS D&B 2023)
- `elliptic2.md` — Elliptic2 (Bellei et al., KDD MLF 2024)

Each card records: provenance and citation, licence and access terms, size and schema,
the temporal range and how the frozen split boundaries were chosen, per-class label
distribution, **which fact families the substrate can and cannot support** (invariant 4,
mirroring `data.availability` in the Hydra config), and known limitations or artefacts.

Neither dataset is redistributed in this repository. Elliptic2 is access-gated and not
redistributable at all.
