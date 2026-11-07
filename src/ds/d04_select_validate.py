"""DS d04: validate the candidate segmentations on business outcomes, choose one, publish it as seg-v1.0.

Outcome tests (outcomes were never model inputs), each on the population the outcome applies to:
    conversion      customers with a closed lead     chi-square on converted        Cramer's V
    deal value      customers with a deal            Kruskal-Wallis                 epsilon squared
    days to close   customers with a closed lead     Kruskal-Wallis                 epsilon squared
    late payment    customers with payments due      chi-square on any late payment Cramer's V
Actionably distinct pairs: share of segment pairs that differ on at least one outcome both significantly
(Bonferroni over all pairwise tests, family alpha 0.01) and by a practical margin (PRACTICAL below).

Selection rule (fixed before the tests were run): among the d03 candidates (the d02 baseline is the benchmark),
best mean rank over stability, Gower silhouette, the four effect sizes, distinct-pair share and robustness to
categorical emphasis (lower of the two sensitivity ARIs). Ties go to stability. The answer key is not used.

Publishes: ds.customer_segments (seg-v1.0, segment ids ordered by size, confidence), ds.segment_validation,
MLflow run d04-select with the champion model registered as REGISTERED_MODEL alias "champion".
"""
import itertools
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from config import DS_REPORTS_DIR, MODEL_VERSION  # noqa: E402
from src.db import connect, query_df, run_sql_file  # noqa: E402
from src.ds import tracking  # noqa: E402
from src.ds.candidates import load_features  # noqa: E402
from src.ds.segmenter import Segmenter, SegmenterPyfunc  # noqa: E402

REGISTERED_MODEL = "veltrix-segmenter"
FAMILY_ALPHA = 0.01
PRACTICAL = {"conversion": 3.0, "late_payment": 2.0, "deal_value_ratio": 1.25, "days_to_close": 7.0}
CRITERIA = ["stability_ari_mean", "silhouette_gower", "v_conversion", "eps2_deal_value", "eps2_days_to_close",
            "v_late_payment", "distinct_pair_share", "emphasis_robustness"]


# --- data ------------------------------------------------------------------------------------------

def load_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = query_df("SELECT candidate_id, model, k, feature_space, mlflow_run_id, metrics FROM ds.candidate_registry")
    reg = reg.join(pd.json_normalize(reg.pop("metrics")))
    labels = query_df("SELECT candidate_id, customer_id, segment FROM ds.segment_candidates") \
        .pivot(index="customer_id", columns="candidate_id", values="segment")
    return reg.set_index("candidate_id"), labels


def load_outcomes() -> pd.DataFrame:
    return query_df("""
        SELECT customer_id, n_closed_leads, converted, n_transactions, deal_value_gbp, med_days_to_close,
               n_payments_due, n_payments_late > 0 AS any_late
        FROM ds.customer_outcomes""").set_index("customer_id")


# --- tests -----------------------------------------------------------------------------------------

def cramers_v(table: pd.DataFrame) -> tuple[float, float, float]:
    chi2, p = chi2_contingency(table, correction=False)[:2]
    return chi2, p, float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))))


def outcome_tests(seg: pd.Series, o: pd.DataFrame) -> list[dict]:
    d = o.join(seg.rename("segment"))
    rows = []
    for name, pop_mask, col, kind in [
        ("conversion", d.n_closed_leads > 0, "converted", "chi2"),
        ("deal_value", d.n_transactions > 0, "deal_value_gbp", "kruskal"),
        ("days_to_close", d.n_closed_leads > 0, "med_days_to_close", "kruskal"),
        ("late_payment", d.n_payments_due > 0, "any_late", "chi2"),
    ]:
        sub = d[pop_mask].dropna(subset=[col])
        if kind == "chi2":
            stat, p, eff = cramers_v(pd.crosstab(sub.segment, sub[col].astype(bool)))
            measure = "cramers_v"
        else:
            groups = [g[col].astype(float).to_numpy() for _, g in sub.groupby("segment") if len(g) > 1]
            stat, p = kruskal(*groups)
            k, n = len(groups), len(sub)
            eff, measure = float((stat - k + 1) / (n - k)), "epsilon_squared"
        rows.append({"outcome": name, "test": "chi-square" if kind == "chi2" else "Kruskal-Wallis",
                     "population": {"conversion": "customers with a closed lead", "deal_value": "customers with a deal",
                                    "days_to_close": "customers with a closed lead",
                                    "late_payment": "customers with payments due"}[name],
                     "n": int(len(sub)), "statistic": float(stat), "p_value": float(p), "effect_size": eff,
                     "effect_measure": measure})
    return rows


def distinct_pairs(seg: pd.Series, o: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Share of segment pairs that differ significantly AND by a practical margin on at least one outcome."""
    d = o.join(seg.rename("segment"))
    closed, deals, due = d[d.n_closed_leads > 0], d[d.n_transactions > 0], d[d.n_payments_due > 0]
    segs = sorted(d.segment.unique())
    pairs = list(itertools.combinations(segs, 2))
    alpha = FAMILY_ALPHA / (len(pairs) * 4)
    rows = []
    for a, b in pairs:
        reasons = []
        ca, cb = closed[closed.segment == a].converted, closed[closed.segment == b].converted
        if abs(ca.mean() - cb.mean()) * 100 >= PRACTICAL["conversion"] and \
                chi2_contingency(pd.crosstab(np.r_[np.zeros(len(ca)), np.ones(len(cb))],
                                             np.r_[ca.to_numpy(), cb.to_numpy()].astype(bool)))[1] < alpha:
            reasons.append("conversion")
        la, lb = due[due.segment == a].any_late, due[due.segment == b].any_late
        if len(la) and len(lb) and abs(la.mean() - lb.mean()) * 100 >= PRACTICAL["late_payment"] and \
                chi2_contingency(pd.crosstab(np.r_[np.zeros(len(la)), np.ones(len(lb))],
                                             np.r_[la.to_numpy(), lb.to_numpy()].astype(bool)))[1] < alpha:
            reasons.append("late_payment")
        va, vb = deals[deals.segment == a].deal_value_gbp, deals[deals.segment == b].deal_value_gbp
        if len(va) > 1 and len(vb) > 1:
            ratio = max(va.median(), vb.median()) / max(min(va.median(), vb.median()), 1)
            if ratio >= PRACTICAL["deal_value_ratio"] and mannwhitneyu(va, vb).pvalue < alpha:
                reasons.append("deal_value")
        ta, tb = closed[closed.segment == a].med_days_to_close.dropna(), closed[closed.segment == b].med_days_to_close.dropna()
        if abs(ta.median() - tb.median()) >= PRACTICAL["days_to_close"] and mannwhitneyu(ta, tb).pvalue < alpha:
            reasons.append("days_to_close")
        rows.append({"segment_a": a, "segment_b": b, "distinct": bool(reasons), "differs_on": ", ".join(reasons)})
    table = pd.DataFrame(rows)
    return float(table.distinct.mean()), table


# --- selection -------------------------------------------------------------------------------------

def scorecard(reg: pd.DataFrame, labels: pd.DataFrame, o: pd.DataFrame):
    rows, tests, pair_tables = [], {}, {}
    for cid in reg.index:
        seg = labels[cid]
        t = outcome_tests(seg, o)
        share, pairs = distinct_pairs(seg, o)
        eff = {r["outcome"]: r["effect_size"] for r in t}
        sens = [reg.loc[cid, c] for c in reg.columns if c.startswith("sensitivity_ari") and pd.notna(reg.loc[cid, c])]
        rows.append({"candidate_id": cid, "model": reg.loc[cid, "model"], "k": int(reg.loc[cid, "k"]),
                     "eligible": not cid.startswith("baseline"),
                     "stability_ari_mean": reg.loc[cid, "stability_ari_mean"],
                     "silhouette_gower": reg.loc[cid, "silhouette_gower"],
                     "v_conversion": eff["conversion"], "eps2_deal_value": eff["deal_value"],
                     "eps2_days_to_close": eff["days_to_close"], "v_late_payment": eff["late_payment"],
                     "distinct_pair_share": share,
                     "emphasis_robustness": min(sens) if sens else np.nan,
                     "min_share": reg.loc[cid, "min_share"]})
        tests[cid], pair_tables[cid] = t, pairs
    card = pd.DataFrame(rows).set_index("candidate_id")
    elig = card[card.eligible]
    ranks = elig[CRITERIA].rank(ascending=False, method="min")
    card["mean_rank"] = ranks.mean(axis=1)
    best = card[card.eligible].sort_values(["mean_rank", "stability_ari_mean"], ascending=[True, False]).index[0]
    return card, tests, pair_tables, best


def agreement_matrix(labels: pd.DataFrame) -> pd.DataFrame:
    ids = list(labels.columns)
    return pd.DataFrame([[adjusted_rand_score(labels[a], labels[b]) for b in ids] for a in ids], index=ids, columns=ids)


# --- publishing ------------------------------------------------------------------------------------

def size_order_map(raw: np.ndarray) -> dict:
    """Raw cluster number -> segment id 1..k, 1 = largest (dashboard sorts segments by customer count)."""
    order = pd.Series(raw).value_counts().index
    return {int(r): i + 1 for i, r in enumerate(order)}


def write_segments(df: pd.DataFrame, seg_ids: np.ndarray, conf: np.ndarray, candidate_id: str) -> None:
    run_sql_file("ds/ds04_customer_segments.sql")
    out = pd.DataFrame({"customer_id": df.customer_id.to_numpy(), "model_version": MODEL_VERSION,
                        "segment_id": seg_ids, "segment_name": [f"Segment {s}" for s in seg_ids],
                        "confidence": np.round(conf, 4), "candidate_id": candidate_id})
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM ds.customer_segments WHERE model_version = %s", (MODEL_VERSION,))
        with cur.copy("COPY ds.customer_segments (customer_id, model_version, segment_id, segment_name, confidence, "
                      "candidate_id) FROM STDIN WITH (FORMAT csv)") as cp:
            cp.write(buf.getvalue())


def write_validation(tests: dict) -> None:
    rows = [(MODEL_VERSION, cid, r["outcome"], r["test"], r["population"], r["n"], r["statistic"], r["p_value"],
             r["effect_size"], r["effect_measure"]) for cid, t in tests.items() for r in t]
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM ds.segment_validation WHERE model_version = %s", (MODEL_VERSION,))
        cur.executemany("INSERT INTO ds.segment_validation VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", rows)


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def plot_agreement(ari: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(ari.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ari)), ari.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(ari)), ari.index)
    for i in range(len(ari)):
        for j in range(len(ari)):
            ax.text(j, i, f"{ari.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if ari.values[i, j] > 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Adjusted Rand index")
    ax.set_title("How much the candidate segmentations agree")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(card, tests, pair_tables, ari, best, crosstab, path) -> None:
    fmt = card.copy()
    lines = [f"# Segment validation and model choice ({MODEL_VERSION})", "",
             "Outcomes were never model inputs. Answer key not used.", "",
             f"**Chosen: `{best}`** (best mean rank over {', '.join(CRITERIA)}; the d02 baseline is the benchmark).", "",
             "## Scorecard", "", md_table(fmt.round(3)), "",
             "## Outcome tests", ""]
    for cid, t in tests.items():
        lines += [f"### {cid}", "", md_table(pd.DataFrame(t).round({"statistic": 1, "effect_size": 3}), index=False), ""]
    lines += ["## Actionably distinct segment pairs (chosen candidate)", "",
              f"Practical margins: {PRACTICAL}; Bonferroni family alpha {FAMILY_ALPHA}.", "",
              md_table(pair_tables[best], index=False), "",
              "## Agreement between candidates (ARI)", "", md_table(ari.round(2)), "",
              "## Chosen vs runner-up: where customers go", "", md_table(crosstab), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    reg, labels = load_candidates()
    o = load_outcomes()
    print(f"Validating {len(reg)} candidates on {len(labels):,} customers")

    card, tests, pair_tables, best = scorecard(reg, labels, o)
    ari = agreement_matrix(labels)
    runner_up = card[card.eligible].drop(best).sort_values("mean_rank").index[0]
    crosstab = pd.crosstab(labels[best].rename(best), labels[runner_up].rename(runner_up))

    show = ["k", "stability_ari_mean", "silhouette_gower", "v_conversion", "eps2_deal_value",
            "eps2_days_to_close", "v_late_payment", "distinct_pair_share", "emphasis_robustness", "mean_rank"]
    print(card[show].round(3).to_string())
    print(f"\nAgreement (ARI):\n{ari.round(2).to_string()}")
    print(f"\nChosen: {best} (runner-up {runner_up})")
    print(f"{best} vs {runner_up} (rows = {best}):\n{crosstab.to_string()}")

    # Champion: reload the chosen d03 model, renumber segments by size, add confidence
    run_id = reg.loc[best, "mlflow_run_id"]
    tracking.setup()
    src_seg = mlflow.pyfunc.load_model(f"runs:/{run_id}/segmenter_{reg.loc[best, 'model']}") \
        .unwrap_python_model().segmenter
    df, _ = load_features()
    raw = Segmenter(src_seg.pre, src_seg.model, src_seg.kind, src_seg.emphasis).predict(df)
    if not (raw == labels.loc[df.customer_id, best].to_numpy()).all():
        raise RuntimeError("Reloaded model does not reproduce the stored candidate labels")
    champion = Segmenter(src_seg.pre, src_seg.model, src_seg.kind, src_seg.emphasis, label_map=size_order_map(raw))
    seg_ids = champion.predict(df)
    conf = champion.confidence(df)

    DS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    card.to_csv(DS_REPORTS_DIR / "d04_scorecard.csv")
    plot_agreement(ari, DS_REPORTS_DIR / "d04_candidate_agreement.png")
    write_report(card, tests, pair_tables, ari, best, crosstab, DS_REPORTS_DIR / "d04_validation_report.md")

    with mlflow.start_run(run_name="d04-select") as run:
        mlflow.set_tags({"step": "d04", "model_version": MODEL_VERSION, "chosen_candidate": best,
                         "source_run_id": run_id})
        mlflow.log_params({"chosen_candidate": best, "runner_up": runner_up, "k": int(card.loc[best, "k"]),
                           "criteria": ", ".join(CRITERIA), "practical_margins": str(PRACTICAL),
                           "family_alpha": FAMILY_ALPHA})
        mlflow.log_metrics({c: float(card.loc[best, c]) for c in CRITERIA + ["min_share"]})
        mlflow.log_metrics({"median_confidence": float(np.median(conf)),
                            "low_confidence_share": float((conf < 0.1).mean())})
        for name in ("d04_scorecard.csv", "d04_candidate_agreement.png", "d04_validation_report.md"):
            mlflow.log_artifact(str(DS_REPORTS_DIR / name))
        with tracking.staged_code_paths() as code_paths:
            info = mlflow.pyfunc.log_model(name="segmenter", python_model=SegmenterPyfunc(champion),
                                           code_paths=code_paths, input_example=df.head(5),
                                           registered_model_name=REGISTERED_MODEL)
        client = mlflow.MlflowClient()
        version = info.registered_model_version
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", version)
        client.set_model_version_tag(REGISTERED_MODEL, version, "model_version", MODEL_VERSION)
        client.set_model_version_tag(REGISTERED_MODEL, version, "candidate_id", best)

    write_segments(df, seg_ids, conf, best)
    write_validation(tests)

    sizes = pd.Series(seg_ids).value_counts().sort_index()
    print(f"\nPublished {MODEL_VERSION}: {len(seg_ids):,} customers in {len(sizes)} segments "
          f"({', '.join(f'{s}: {n:,}' for s, n in sizes.items())})")
    print(f"Median confidence {np.median(conf):.2f}; {(conf < 0.1).mean():.1%} below 0.1")
    print(f"Registered {REGISTERED_MODEL} v{version} as 'champion'; MLflow run {run.info.run_id}")


if __name__ == "__main__":
    main()
