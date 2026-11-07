"""DS d99: external evaluation against the restricted answer key. Run once, after everything is fixed.

This is the ONLY code that reads data/raw/veltrix_medium/_restricted/. Nothing here feeds back into features,
model choice or names; results are logged to MLflow tagged external_evaluation=true.

1. Segment recovery: ARI and NMI of every candidate and of the published seg-v1.0 against the true personas
   (project target ARI >= 0.5), a segment x persona table, purity, and whether each blind d05 name matches the
   segment's majority persona.
2. Deduplication: a merge is correct when both records resolve to the same true entity
   (entity = duplicate_of if set, else the record itself). Precision by method (hash, profile) and for the
   unmerged probable_duplicate flags; recall over all true duplicate records.
"""
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import DS_REPORTS_DIR, MODEL_VERSION, RAW_DIR
from src.db import query_df
from src.ds import tracking

ANSWER_KEY = RAW_DIR / "_restricted" / "_answer_key_customer_persona.parquet"
TARGET_ARI = 0.5
# blind d05 names -> the persona each name claims (for the name check only)
NAME_TO_PERSONA = {
    "Window shoppers": "window_shopper", "Family home movers": "home_mover_family",
    "First-time buyers": "first_time_buyer", "Young renters": "young_renter",
    "Buy-to-let investors": "btl_investor", "Overseas investors": "overseas_investor",
    "Downsizers": "downsizer", "Corporate relocations": "corporate_relocation",
}


def load_key() -> pd.DataFrame:
    key = pd.read_parquet(ANSWER_KEY)
    key["entity"] = key.duplicate_of.fillna(key.customer_id)
    return key.set_index("customer_id")


def segment_recovery(key: pd.DataFrame):
    cands = query_df("SELECT candidate_id, customer_id, segment FROM ds.segment_candidates") \
        .pivot(index="customer_id", columns="candidate_id", values="segment")
    published = query_df("SELECT customer_id, segment_id, segment_name FROM ds.customer_segments "
                         "WHERE model_version = %s", (MODEL_VERSION,)).set_index("customer_id")
    persona = key.true_persona.reindex(published.index)
    if persona.isna().any():
        raise ValueError(f"{persona.isna().sum()} clean customers missing from the answer key")

    rows = []
    for cid in cands.columns:
        lab = cands[cid].reindex(published.index)
        rows.append({"labels": cid, "k": lab.nunique(), "ari": adjusted_rand_score(persona, lab),
                     "nmi": normalized_mutual_info_score(persona, lab)})
    rows.append({"labels": f"{MODEL_VERSION} (published)", "k": published.segment_id.nunique(),
                 "ari": adjusted_rand_score(persona, published.segment_id),
                 "nmi": normalized_mutual_info_score(persona, published.segment_id)})
    scores = pd.DataFrame(rows).set_index("labels").sort_values("ari", ascending=False)

    table = pd.crosstab(published.segment_name, persona)
    order = published.groupby("segment_name").segment_id.first().sort_values().index
    table = table.loc[order]
    names = pd.DataFrame({
        "segment_id": published.groupby("segment_name").segment_id.first().loc[order],
        "customers": table.sum(axis=1),
        "majority_persona": table.idxmax(axis=1),
        "purity": table.max(axis=1) / table.sum(axis=1),
        "persona_recall": [table.loc[s, table.loc[s].idxmax()] / table[table.loc[s].idxmax()].sum() for s in order],
    })
    names["name_claims"] = names.index.map(NAME_TO_PERSONA)
    names["name_matches"] = names.name_claims == names.majority_persona
    return scores, table, names


def dedup_quality(key: pd.DataFrame) -> pd.DataFrame:
    merges = query_df("SELECT duplicate_id, survivor_id, method FROM ref.customer_merge")
    probable = query_df("SELECT customer_id AS duplicate_id, candidate_of AS survivor_id FROM ref.probable_duplicates")
    ent = key.entity

    def correct(df):
        return (ent.reindex(df.duplicate_id).to_numpy() == ent.reindex(df.survivor_id).to_numpy())

    merges["correct"] = correct(merges)
    probable["correct"] = correct(probable)
    merges["group"] = np.where(merges.method == "profile", "profile", "hash")
    true_dups = key[key.duplicate_of.notna()]
    caught = merges[merges.correct]
    # a true duplicate is recovered when it sits in a correct merge pair (as the merged or surviving record)
    recovered = set(caught.duplicate_id) | set(caught.survivor_id)
    rows = []
    for g, d in merges.groupby("group"):
        rows.append({"set": f"merged ({g})", "pairs": len(d), "correct": int(d.correct.sum()),
                     "precision": d.correct.mean()})
    rows.append({"set": "merged (all)", "pairs": len(merges), "correct": int(merges.correct.sum()),
                 "precision": merges.correct.mean()})
    rows.append({"set": "flagged probable (not merged)", "pairs": len(probable), "correct": int(probable.correct.sum()),
                 "precision": probable.correct.mean() if len(probable) else np.nan})
    out = pd.DataFrame(rows).set_index("set")
    out["recall_of_true_duplicates"] = np.nan
    out.loc["merged (all)", "recall_of_true_duplicates"] = len(set(true_dups.index) & recovered) / len(true_dups)
    out.attrs["true_duplicates"] = len(true_dups)
    out.attrs["hash_findable"] = int(merges[merges.group == "hash"].correct.sum())
    return out


def fmt_table(df: pd.DataFrame) -> str:
    d = df.reset_index()
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns)
    body = "\n".join("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
                     for r in d.itertuples(index=False))
    return head + "\n" + body


def main() -> None:
    print(f"External evaluation (answer key read for the first and only time: {ANSWER_KEY.name})")
    key = load_key()
    scores, table, names = segment_recovery(key)
    dedup = dedup_quality(key)
    seg_ari = float(scores.loc[f"{MODEL_VERSION} (published)", "ari"])

    lines = [f"# External evaluation against the answer key ({MODEL_VERSION})", "",
             "Run once after the model, names and test design were fixed. Nothing here fed back into the model.", "",
             f"## Segment recovery (target ARI >= {TARGET_ARI})", "", fmt_table(scores.round(3)), "",
             "### Published segments vs true personas", "", fmt_table(table), "",
             "### Blind names vs majority persona", "", fmt_table(names.round(3)), "",
             f"## Deduplication ({dedup.attrs['true_duplicates']:,} true duplicate records)", "",
             fmt_table(dedup.round(3)), ""]
    DS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (DS_REPORTS_DIR / "d99_external_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    scores.to_csv(DS_REPORTS_DIR / "d99_ari.csv")
    table.to_csv(DS_REPORTS_DIR / "d99_segment_vs_persona.csv")

    tracking.setup()
    with mlflow.start_run(run_name="d99-external-eval"):
        mlflow.set_tags({"step": "d99", "external_evaluation": "true", "model_version": MODEL_VERSION})
        mlflow.log_metrics({f"ari_{i.split(' ')[0]}": float(r.ari) for i, r in scores.iterrows()})
        mlflow.log_metrics({"ari_published": seg_ari, "nmi_published": float(scores.loc[f"{MODEL_VERSION} (published)", "nmi"]),
                            "names_matching": int(names.name_matches.sum()), "mean_purity": float(names.purity.mean()),
                            "dedup_precision_all": float(dedup.loc["merged (all)", "precision"]),
                            "dedup_recall": float(dedup.loc["merged (all)", "recall_of_true_duplicates"]),
                            "probable_flag_precision": float(dedup.loc["flagged probable (not merged)", "precision"])})
        for name in ("d99_external_evaluation.md", "d99_ari.csv", "d99_segment_vs_persona.csv"):
            mlflow.log_artifact(str(DS_REPORTS_DIR / name))

    print("\nSegment recovery:\n" + scores.round(3).to_string())
    print(f"\n{MODEL_VERSION} ARI {seg_ari:.3f} -> target {TARGET_ARI} {'MET' if seg_ari >= TARGET_ARI else 'NOT met'}")
    print("\nSegments vs personas:\n" + names[["segment_id", "customers", "majority_persona", "purity", "persona_recall",
                                              "name_matches"]].round(3).to_string())
    print(f"\nDeduplication ({dedup.attrs['true_duplicates']:,} true duplicates):\n" + dedup.round(3).to_string())


if __name__ == "__main__":
    main()
