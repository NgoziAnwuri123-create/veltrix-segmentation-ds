"""Shared helpers for candidate segmentations: data loading and saving labels to ds.segment_candidates."""
import io
import json

import numpy as np
import pandas as pd

from config import FEATURE_VERSION, PROCESSED_DIR
from src.db import connect, query_df, run_sql_file

FEATURES_PARQUET = PROCESSED_DIR / f"customer_features_{FEATURE_VERSION}.parquet"


def load_features() -> tuple[pd.DataFrame, np.ndarray]:
    """fv1 features and the training mask (probable duplicates are scored, not trained on)."""
    df = pd.read_parquet(FEATURES_PARQUET)
    return df, ~df.probable_duplicate.astype(bool).to_numpy()


def segment_outcomes(customer_ids: pd.Series, labels: np.ndarray) -> pd.DataFrame:
    """Business outcomes by segment, for a first look (formal tests are in d04)."""
    o = query_df("SELECT customer_id, converted, conversion_rate, deal_value_gbp, late_payment_rate, "
                 "med_days_to_close FROM ds.customer_outcomes")
    d = pd.DataFrame({"customer_id": customer_ids.to_numpy(), "segment": labels}).merge(o, on="customer_id")
    g = d.groupby("segment")
    out = pd.DataFrame({
        "customers": g.size(),
        "share": g.size() / len(d),
        "converted_pct": 100 * g.converted.mean(),
        "deal_value_share": g.deal_value_gbp.sum() / d.deal_value_gbp.sum(),
        "late_payment_rate": g.late_payment_rate.mean(),
        "med_days_to_close": g.med_days_to_close.median(),
    })
    return out.round(3)


def save_candidate(candidate_id: str, model: str, k: int, feature_space: str, run_id: str,
                   metrics: dict, customer_ids: pd.Series, labels: np.ndarray) -> None:
    run_sql_file("ds/ds03_segment_candidates.sql")
    buf = io.StringIO()
    pd.DataFrame({"candidate_id": candidate_id, "customer_id": customer_ids.to_numpy(), "segment": labels}) \
        .to_csv(buf, index=False, header=False)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM ds.candidate_registry WHERE candidate_id = %s", (candidate_id,))
        cur.execute("INSERT INTO ds.candidate_registry (candidate_id, model, k, feature_space, mlflow_run_id, metrics) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (candidate_id, model, k, feature_space, run_id, json.dumps(metrics)))
        with cur.copy("COPY ds.segment_candidates (candidate_id, customer_id, segment) FROM STDIN WITH (FORMAT csv)") as cp:
            cp.write(buf.getvalue())
