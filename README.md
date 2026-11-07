# Veltrix customer segmentation: data science workflow

Customer segmentation for **Veltrix**, a fictional UK estate agency and new-homes developer (synthetic data,
no real people). This repository holds the data science workflow from model benchmarking to production scoring:
choosing a segmentation, validating it on business outcomes, naming and profiling the segments, designing a
campaign test, and monitoring drift. Experiments are tracked in **MLflow**; data lives in **PostgreSQL**.

## Workflow

| Step | Module | What it does |
|---|---|---|
| d02 | `src/ds/d02_baseline.py` | Naive k-means benchmark (k = 5-12) on raw one-hot features |
| d03 | `src/ds/d03_candidates.py` | Curated candidates: k-means, **k-prototypes**, Gaussian mixture; same sweep and scorecard |
| d04 | `src/ds/d04_select_validate.py` | Outcome tests, selection rule, publishes `seg-v1.0`, registers the MLflow `champion` model |
| d05 | `src/ds/d05_segment_stats.py` | Segment KPIs in SQL, index-vs-average profiles, evidence-based segment names |
| d06 | `src/ds/d06_test_design.py` | Tailored-vs-generic campaign test: sizing, hash randomisation, A/A backtest, monitoring view |
| d07 | `src/ds/d07_score.py` | Monthly scoring with the champion model and drift monitoring (segment size, PSI, migration) |
| d99 | `src/ds/d99_external_eval.py` | One-off evaluation against the restricted answer key (run last; outputs not published) |

Shared modules: `feature_spec.py` (feature dictionary), `preprocess.py` (imputation, transforms, scaling learned
in `fit`), `segmenter.py` (preprocessor + model behind one `predict`, logged as an MLflow pyfunc), `evaluate.py`
and `gower.py` (label-free scorecard), `candidates.py`, `tracking.py`. SQL is kept separate in `sql/ds/`.

## Method highlights

- **Label-free model choice.** Every candidate gets the same scorecard: bootstrap stability (ARI between refits),
  Gower silhouette on a common mixed-type space, segment-size floor (2%), and robustness to categorical weighting.
  The selection rule was fixed before outcome tests were run. The answer key is read only by d99.
- **Validated on outcomes the model never saw.** Conversion, deal value, days to close and late payment are kept
  in a separate outcome table; features that grow as a lead progresses were excluded as leakage.
- **Evidence-based naming.** Segment names come from index-vs-average profiles, not from the dataset documentation.
- **Honest test design.** Sizing showed a one-segment, 6-week enquiry test could only detect 43-140% lifts, so
  the test compares segment-tailored with generic campaigns across segments over 12 weeks (MDE 16.6%).
- **Reproducible production path.** Features can be rebuilt as of any date (`AS_OF`); the champion model loaded by
  alias reproduces the published segments exactly, and a negative-control test confirms the drift alerts fire.

## Results (internal, label-free)

| Candidate | k | Stability | Gower silhouette | Distinct segment pairs |
|---|---|---|---|---|
| baseline k-means (d02) | 8 | 0.84 | 0.153 | 100% |
| Gaussian mixture | 7 | 0.81 | 0.155 | 71% |
| curated k-means | 9 | 0.84 | 0.157 | 97% |
| **k-prototypes (seg-v1.0)** | **8** | **0.91** | **0.163** | **100%** |

seg-v1.0 segments: Window shoppers (23%), Family home movers (18%), First-time buyers (16%), Young renters (16%),
Buy-to-let investors (11%), Overseas investors (7%), Downsizers (7%), Corporate relocations (2%).
Details: `reports/ds/d04_validation_report.md`, `reports/ds/d05_segment_profiles.md`,
`reports/ds/d06_test_design.md`.

## Running it

Prerequisites, produced by the upstream data steps (not in this repository):
- PostgreSQL with the cleaned `clean.*` schema and the `ref.*` reference tables
- `ds.customer_features` / `ds.customer_outcomes` and the snapshot `data/processed/customer_features_fv1.parquet`

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\...
cp .env.example .env            # set the PostgreSQL connection
python run_pipeline.py          # d02 -> d07
python run_pipeline.py --list   # individual steps
python -m src.ds.d07_score refresh 2026-11-01   # one production refresh
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

d99 needs the restricted answer key at `data/raw/veltrix_medium/_restricted/` and is never run by default.
