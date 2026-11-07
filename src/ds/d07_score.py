"""DS d07: score customers with the champion model and monitor drift (monthly refresh).

refresh(as_of):
    1. rebuild ds.customer_features as of the date (sql/ds/ds01_model_features.sql, AS_OF)
    2. score every customer with REGISTERED_MODEL@champion: segment id + assignment confidence
    3. store the scores in ds.segment_score_history
    4. on the latest date only: upsert ds.customer_segments (new customers added, changed segments updated)
    5. drift checks against the reference = the training snapshot (fv1) scored by the same model:
         segment_size_change    |share / reference share - 1|    watch > 0.10, alert > 0.20 (project plan)
         feature_psi            PSI per model feature            watch > 0.10, alert > 0.20
         migration              share changing segment vs the previous refresh   watch > 0.10, alert > 0.15
         low_confidence_change  rise in share with confidence < 0.1              watch > 0.025, alert > 0.05
       Retrain recommended on any size, migration or confidence alert, or when >= 10% of features alert.
    6. ds.monitoring_drift, ds.monitoring_runs, MLflow run.

    python run_pipeline.py score                  backtest: monthly refreshes BACKTEST_DATES (ends on the latest)
    python -m src.ds.d07_score refresh 2026-11-01  one production refresh
"""
import io
import sys
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import DATA_TO, DS_REPORTS_DIR, MODEL_VERSION  # noqa: E402
from src.db import connect, query_df, run_sql_file  # noqa: E402
from src.ds import tracking  # noqa: E402
from src.ds.candidates import load_features  # noqa: E402
from src.ds.feature_spec import MODEL_FEATURES  # noqa: E402

REGISTERED_MODEL = "veltrix-segmenter"
LATEST_AS_OF = DATA_TO + timedelta(days=1)
BACKTEST_DATES = [date(2026, 7, 1), date(2026, 8, 1), date(2026, 9, 1), LATEST_AS_OF]
LOW_CONFIDENCE = 0.1
THRESHOLDS = {  # metric: (watch, alert)
    "segment_size_change": (0.10, 0.20),
    "feature_psi": (0.10, 0.20),
    "migration": (0.10, 0.15),
    "low_confidence_change": (0.025, 0.05),
}
FEATURE_ALERT_SHARE = 0.10
NUM_FEATURES = [f.name for f in MODEL_FEATURES if f.kind != "cat"]
CAT_FEATURES = [f.name for f in MODEL_FEATURES if f.kind == "cat"]


# --- model and scoring ----------------------------------------------------------------------------

def load_champion():
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@champion")
    version = mlflow.MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, "champion")
    return model.unwrap_python_model().segmenter, version


def score(segmenter, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"customer_id": df.customer_id.to_numpy(), "segment_id": segmenter.predict(df),
                         "confidence": segmenter.confidence(df)})


def build_features(as_of: date) -> pd.DataFrame:
    run_sql_file("ds/ds01_model_features.sql", params={"AS_OF": as_of.isoformat()})
    return query_df("SELECT * FROM ds.customer_features ORDER BY customer_id")


# --- drift -----------------------------------------------------------------------------------------

def status(metric: str, value: float) -> str:
    watch, alert = THRESHOLDS[metric]
    v = abs(value)
    return "alert" if v > alert else "watch" if v > watch else "ok"


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """PSI on reference deciles; missing values form their own bin (shares are NULL without activity)."""
    r, c = ref.astype(float).to_numpy(), cur.astype(float).to_numpy()
    edges = np.unique(np.nanquantile(r, np.linspace(0, 1, bins + 1))[1:-1])

    def dist(x):
        idx = np.where(np.isnan(x), len(edges) + 1, np.searchsorted(edges, np.nan_to_num(x), side="right"))
        return np.bincount(idx, minlength=len(edges) + 2) / len(x)

    return psi_from_dist(dist(r), dist(c))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    r = ref.astype(object).fillna("Unknown").value_counts(normalize=True)
    c = cur.astype(object).fillna("Unknown").value_counts(normalize=True)
    levels = r.index.union(c.index)
    return psi_from_dist(r.reindex(levels, fill_value=0).to_numpy(), c.reindex(levels, fill_value=0).to_numpy())


def psi_from_dist(r: np.ndarray, c: np.ndarray, eps: float = 1e-4) -> float:
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_checks(ref_df, ref_scores, cur_df, cur_scores, prev_scores) -> pd.DataFrame:
    rows = []
    ref_share = ref_scores.segment_id.value_counts(normalize=True)
    cur_share = cur_scores.segment_id.value_counts(normalize=True).reindex(ref_share.index, fill_value=0)
    for s in sorted(ref_share.index):
        change = cur_share[s] / ref_share[s] - 1
        rows.append(("segment_size_change", f"segment {s}", change, ref_share[s], "segment_size_change"))
    for f in NUM_FEATURES:
        rows.append(("feature_psi", f, psi_numeric(ref_df[f], cur_df[f]), None, "feature_psi"))
    for f in CAT_FEATURES:
        rows.append(("feature_psi", f, psi_categorical(ref_df[f], cur_df[f]), None, "feature_psi"))
    if prev_scores is not None:
        m = cur_scores.merge(prev_scores, on="customer_id", suffixes=("", "_prev"))
        rows.append(("migration", "all", float((m.segment_id != m.segment_id_prev).mean()), None, "migration"))
    ref_low = float((ref_scores.confidence < LOW_CONFIDENCE).mean())
    cur_low = float((cur_scores.confidence < LOW_CONFIDENCE).mean())
    rows.append(("low_confidence_change", "all", cur_low - ref_low, ref_low, "low_confidence_change"))
    out = pd.DataFrame(rows, columns=["metric", "item", "value", "reference", "rule"])
    out["threshold"] = out.rule.map(lambda m: THRESHOLDS[m][1])
    out["status"] = [status(m, v) for m, v in zip(out.rule, out.value)]
    return out.drop(columns="rule")


def retrain_decision(drift: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons = []
    for metric in ("segment_size_change", "migration", "low_confidence_change"):
        hit = drift[(drift.metric == metric) & (drift.status == "alert")]
        reasons += [f"{metric} {r.item}: {r.value:+.1%}" for r in hit.itertuples()]
    feats = drift[drift.metric == "feature_psi"]
    alert_share = float((feats.status == "alert").mean())
    if alert_share >= FEATURE_ALERT_SHARE:
        top = feats.nlargest(3, "value")
        reasons.append(f"{alert_share:.0%} of features drifted (PSI > 0.2), e.g. "
                       + ", ".join(f"{r.item} {r.value:.2f}" for r in top.itertuples()))
    return bool(reasons), reasons


# --- persistence -------------------------------------------------------------------------------------

def write_history(as_of: date, scores: pd.DataFrame) -> None:
    buf = io.StringIO()
    scores.assign(as_of=as_of, model_version=MODEL_VERSION)[
        ["as_of", "model_version", "customer_id", "segment_id", "confidence"]].to_csv(buf, index=False, header=False)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM ds.segment_score_history WHERE as_of = %s AND model_version = %s", (as_of, MODEL_VERSION))
        with cur.copy("COPY ds.segment_score_history FROM STDIN WITH (FORMAT csv)") as cp:
            cp.write(buf.getvalue())


def upsert_current_segments(scores: pd.DataFrame, candidate_id: str) -> tuple[int, int]:
    """Bring ds.customer_segments up to date. Returns (new customers, customers whose segment changed)."""
    existing = query_df("SELECT customer_id, segment_id FROM ds.customer_segments WHERE model_version = %s",
                        (MODEL_VERSION,)).set_index("customer_id").segment_id
    names = query_df("SELECT segment_id, segment_name FROM ds.segment_names WHERE model_version = %s",
                     (MODEL_VERSION,)).set_index("segment_id").segment_name
    s = scores.set_index("customer_id")
    new = int((~s.index.isin(existing.index)).sum())
    common = s.index.intersection(existing.index)
    changed = int((s.loc[common, "segment_id"] != existing.loc[common]).sum())
    out = scores.assign(model_version=MODEL_VERSION, segment_name=scores.segment_id.map(names),
                        confidence=scores.confidence.round(4), candidate_id=candidate_id)
    buf = io.StringIO()
    out[["customer_id", "model_version", "segment_id", "segment_name", "confidence", "candidate_id"]] \
        .to_csv(buf, index=False, header=False)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE incoming (LIKE ds.customer_segments INCLUDING DEFAULTS) ON COMMIT DROP")
        with cur.copy("COPY incoming (customer_id, model_version, segment_id, segment_name, confidence, candidate_id) "
                      "FROM STDIN WITH (FORMAT csv)") as cp:
            cp.write(buf.getvalue())
        # existing customers keep their candidate_id; only the segment, name, confidence and timestamp move
        cur.execute("""INSERT INTO ds.customer_segments (customer_id, model_version, segment_id, segment_name,
                                                         confidence, candidate_id)
                       SELECT customer_id, model_version, segment_id, segment_name, confidence, candidate_id
                       FROM incoming
                       ON CONFLICT (customer_id, model_version) DO UPDATE
                       SET segment_id = EXCLUDED.segment_id, segment_name = EXCLUDED.segment_name,
                           confidence = EXCLUDED.confidence,
                           assigned_at = CASE WHEN ds.customer_segments.segment_id <> EXCLUDED.segment_id
                                              THEN now() ELSE ds.customer_segments.assigned_at END""")
    return new, changed


def write_monitoring(as_of: date, drift: pd.DataFrame, n_scored, n_new, n_changed, retrain, reasons, run_id) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM ds.monitoring_drift WHERE as_of = %s AND model_version = %s", (as_of, MODEL_VERSION))
        cur.executemany("INSERT INTO ds.monitoring_drift VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        [(as_of, MODEL_VERSION, r.metric, r.item, float(r.value),
                          None if pd.isna(r.reference) else float(r.reference), float(r.threshold), r.status)
                         for r in drift.itertuples()])
        cur.execute("DELETE FROM ds.monitoring_runs WHERE as_of = %s AND model_version = %s", (as_of, MODEL_VERSION))
        cur.execute("INSERT INTO ds.monitoring_runs (as_of, model_version, customers_scored, new_customers, "
                    "segment_changes, alerts, watches, retrain_recommended, reasons, mlflow_run_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (as_of, MODEL_VERSION, n_scored, n_new, n_changed, int((drift.status == "alert").sum()),
                     int((drift.status == "watch").sum()), retrain, "; ".join(reasons) or None, run_id))


# --- refresh -----------------------------------------------------------------------------------------

def previous_scores(as_of: date) -> pd.DataFrame | None:
    prev = query_df("SELECT max(as_of) AS d FROM ds.segment_score_history WHERE as_of < %s AND model_version = %s",
                    (as_of, MODEL_VERSION)).d.iloc[0]
    if prev is None or pd.isna(prev):
        return None
    return query_df("SELECT customer_id, segment_id FROM ds.segment_score_history WHERE as_of = %s AND model_version = %s",
                    (prev, MODEL_VERSION))


def refresh(as_of: date, segmenter, version, ref_df, ref_scores) -> dict:
    print(f"\n--- refresh as of {as_of} ---", flush=True)
    cur_df = build_features(as_of)
    cur_scores = score(segmenter, cur_df)
    prev = previous_scores(as_of)
    drift = drift_checks(ref_df, ref_scores, cur_df, cur_scores, prev)
    retrain, reasons = retrain_decision(drift)
    write_history(as_of, cur_scores)
    if as_of >= LATEST_AS_OF:
        n_new, n_changed = upsert_current_segments(cur_scores, version.tags.get("candidate_id", "champion"))
    elif prev is not None:
        m = cur_scores.merge(prev, on="customer_id", suffixes=("", "_prev"))
        n_new, n_changed = len(cur_scores) - len(m), int((m.segment_id != m.segment_id_prev).sum())
    else:
        n_new, n_changed = len(cur_scores), 0

    with mlflow.start_run(run_name=f"refresh-{as_of}", nested=mlflow.active_run() is not None) as run:
        mlflow.set_tags({"step": "d07", "as_of": str(as_of), "model_version": MODEL_VERSION,
                         "champion_version": version.version})
        sizes = drift[drift.metric == "segment_size_change"]
        mlflow.log_metrics({"customers_scored": len(cur_scores), "new_customers": n_new, "segment_changes": n_changed,
                            "alerts": int((drift.status == "alert").sum()), "watches": int((drift.status == "watch").sum()),
                            "max_abs_segment_size_change": float(sizes.value.abs().max()),
                            "max_feature_psi": float(drift[drift.metric == "feature_psi"].value.max()),
                            "low_confidence_change": float(drift[drift.metric == "low_confidence_change"].value.iloc[0]),
                            "retrain_recommended": int(retrain)})
        write_monitoring(as_of, drift, len(cur_scores), n_new, n_changed, retrain, reasons, run.info.run_id)

    mig = drift[drift.metric == "migration"]
    print(f"  scored {len(cur_scores):,} (new {n_new:,}, changed {n_changed:,}); "
          f"max |size change| {sizes.value.abs().max():.1%}; max PSI "
          f"{drift[drift.metric == 'feature_psi'].value.max():.3f}; "
          + (f"migration {mig.value.iloc[0]:.1%}; " if len(mig) else "")
          + f"alerts {int((drift.status == 'alert').sum())}, watches {int((drift.status == 'watch').sum())}")
    print(f"  retrain recommended: {retrain}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    return {"as_of": as_of, "drift": drift, "retrain": retrain}


def plot_backtest(path) -> None:
    trend = query_df("SELECT as_of, segment_id, segment_name, share, reference_share FROM ds.v_segment_size_trend "
                     "WHERE model_version = %s ORDER BY as_of, segment_id", (MODEL_VERSION,))
    psi = query_df("SELECT as_of, item, value FROM ds.monitoring_drift WHERE model_version = %s AND metric = 'feature_psi'",
                   (MODEL_VERSION,))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for (sid, name), g in trend.groupby(["segment_id", "segment_name"]):
        line, = a1.plot(pd.to_datetime(g.as_of), 100 * (g.share / g.reference_share - 1), marker="o", label=f"{sid}. {name}")
    for y in (-20, 20):
        a1.axhline(y, color="#E8590C", ls="--", lw=1)
    a1.axhline(0, color="grey", lw=0.8)
    a1.set_title("Segment size vs published reference (%); alert beyond ±20%")
    a1.legend(fontsize=8, ncol=2)
    top = psi.groupby("item").value.max().nlargest(8).index
    for item in top:
        g = psi[psi.item == item].sort_values("as_of")
        a2.plot(pd.to_datetime(g.as_of), g.value, marker="o", label=item)
    a2.axhline(0.2, color="#E8590C", ls="--", lw=1)
    a2.axhline(0.1, color="grey", ls=":", lw=1)
    a2.set_title("Feature PSI vs training snapshot (8 most drifted); alert > 0.2")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    argv = argv or []
    dates = [date.fromisoformat(argv[1])] if argv[:1] == ["refresh"] and len(argv) > 1 else BACKTEST_DATES
    run_sql_file("ds/ds07_monitoring.sql")
    tracking.setup()
    segmenter, version = load_champion()
    print(f"Champion: {REGISTERED_MODEL} v{version.version} ({version.tags.get('candidate_id')}, "
          f"{version.tags.get('model_version')})")
    ref_df, _ = load_features()
    ref_scores = score(segmenter, ref_df)
    published = query_df("SELECT customer_id, segment_id FROM ds.customer_segments WHERE model_version = %s",
                         (MODEL_VERSION,)).set_index("customer_id").segment_id
    agree = float((published.reindex(ref_scores.customer_id).to_numpy() == ref_scores.segment_id.to_numpy()).mean())
    print(f"Reference: training snapshot, {len(ref_scores):,} customers; champion reproduces published "
          f"segments for {agree:.2%}")

    with mlflow.start_run(run_name="d07-refresh" if len(dates) == 1 else "d07-backtest"):
        mlflow.set_tags({"step": "d07", "model_version": MODEL_VERSION, "dates": ", ".join(map(str, dates))})
        results = [refresh(d, segmenter, version, ref_df, ref_scores) for d in dates]
        DS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_backtest(DS_REPORTS_DIR / "d07_drift_backtest.png")
        summary = query_df("SELECT * FROM ds.monitoring_runs WHERE model_version = %s ORDER BY as_of", (MODEL_VERSION,))
        summary.to_csv(DS_REPORTS_DIR / "d07_monitoring_runs.csv", index=False)
        for name in ("d07_drift_backtest.png", "d07_monitoring_runs.csv"):
            mlflow.log_artifact(str(DS_REPORTS_DIR / name))
    print("\n" + summary[["as_of", "customers_scored", "new_customers", "segment_changes", "alerts", "watches",
                          "retrain_recommended"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
