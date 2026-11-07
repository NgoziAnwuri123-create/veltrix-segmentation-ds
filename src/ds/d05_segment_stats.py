"""DS d05: segment statistics, index-versus-average profiles and segment names for MODEL_VERSION.

1. ds.segment_stats and ds.segment_channels in SQL (sql/ds/ds05_segment_stats.sql), so DA can reconcile.
2. ds.segment_profile: every feature in the spec, model input or not, per segment against the average.
       numeric  segment mean (median for prices) vs overall; index = 100 x segment / overall
       category share of the segment in each level vs the overall share; index likewise
3. Signature per segment: the most distinctive profile rows (largest |log index|), ignoring rare traits.
4. Names: NAMES below were written from the d05 evidence (signatures, stats, channels), never from the answer
   key. When filled they are applied to ds.customer_segments.segment_name and ds.segment_names.
Outputs: reports/ds/d05_segment_profiles.md, d05_index_heatmap.png, MLflow run "d05-profiles".
"""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import DS_REPORTS_DIR, MODEL_VERSION  # noqa: E402
from src.db import connect, query_df, run_sql_file  # noqa: E402
from src.ds import tracking  # noqa: E402
from src.ds.candidates import load_features  # noqa: E402
from src.ds.feature_spec import FEATURES  # noqa: E402

MIN_PREVALENCE = 0.05      # ignore traits held by < 5% of both the segment and the average customer
MIN_LEVEL_SHARE = 0.15     # a category level must cover >= 15% of the segment to be a signature trait
SIGNATURE_SIZE = 10
HEATMAP_FEATURES = ["price_level_proxy", "share_rent_leads", "is_overseas", "share_night_sessions", "rg_london",
                    "rg_north", "rg_south_west_wales", "share_offplan", "pt_flat", "pt_bungalow", "pt_detached",
                    "dv_mobile", "ev_virtual_tour", "ch_social", "ch_relocation_partner", "tp_school_catchment",
                    "tp_shared_ownership", "tp_rental_yield", "enquiries_per_100_views", "n_sessions"]

# segment_id -> (name, one-line description), written from the d05 evidence pass (signatures, stats, channels).
NAMES: dict[int, tuple[str, str]] = {
    1: ("Window shoppers",
        "Heaviest browsers (28 sessions) who almost never enquire (0.06 enquiries per 100 views); half give no budget."),
    2: ("Family home movers",
        "Chain-bound home movers (79%) buying semis in the South East around £470k; school catchment and chains."),
    3: ("First-time buyers",
        "First-time buyers (81%) using LISAs and shared ownership; flats in the Midlands and North, mobile-first, slow."),
    4: ("Young renters",
        "Under-35 renters (67%) at about £1,400 pcm; mobile and WhatsApp; decide in 9 days; late on rent 14% of the time."),
    5: ("Buy-to-let investors",
        "Portfolio landlords (83%), mostly via limited companies, chasing high-yield Northern stock; repeat enquirers."),
    6: ("Overseas investors",
        "Overseas buyers (95%) of London new-build and off-plan around £640k; WeChat, agents abroad, night browsing."),
    7: ("Downsizers",
        "Older, chain-free downsizers (82%) buying bungalows in the South West and Wales; phone and branch; convert well."),
    8: ("Corporate relocations",
        "Company lets (69%) via relocation partners at about £3,200 pcm; decide in 12 days; best conversion, pay on time."),
}


def load_profile_frame() -> pd.DataFrame:
    df, _ = load_features()
    seg = query_df("SELECT customer_id, segment_id FROM ds.customer_segments WHERE model_version = %s",
                   (MODEL_VERSION,))
    d = df.merge(seg, on="customer_id")
    # a single readable price indicator: stated budget, else viewed price, relative to the intent median
    rent = d.primary_intent.eq("Rent")
    price = np.where(rent, d.rent_budget_pcm_gbp.fillna(d.med_rent_viewed_pcm_gbp),
                     d.budget_gbp.fillna(d.med_asking_viewed_gbp)).astype(float)
    d["price_level_proxy"] = price / pd.Series(price).groupby(d.primary_intent.to_numpy()).transform("median").to_numpy()
    return d


def build_profile(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric = [f for f in FEATURES if f.kind != "cat"] + [type(FEATURES[0])("price_level_proxy", "profile", "price",
                                                                                False, "Price vs intent median")]
    for f in numeric:
        x = d[f.name].astype(float)
        stat = "median" if f.kind == "price" else "mean"
        overall = x.median() if stat == "median" else x.mean()
        by = x.groupby(d.segment_id).median() if stat == "median" else x.groupby(d.segment_id).mean()
        for s, v in by.items():
            rows.append({"segment_id": s, "feature": f.name, "level": "", "group": f.group, "kind": f.kind,
                         "model_input": f.model, "statistic": stat, "segment_value": v, "overall_value": overall})
    for f in (f for f in FEATURES if f.kind == "cat"):
        x = d[f.name].astype(object).fillna("Unknown")
        overall = x.value_counts(normalize=True)
        by = pd.crosstab(d.segment_id, x, normalize="index")
        for s in by.index:
            for level in by.columns:
                rows.append({"segment_id": s, "feature": f.name, "level": str(level), "group": f.group, "kind": "cat",
                             "model_input": f.model, "statistic": "share", "segment_value": by.loc[s, level],
                             "overall_value": overall[level]})
    p = pd.DataFrame(rows)
    p["index"] = np.where(p.overall_value > 0, 100 * p.segment_value / p.overall_value, np.nan)
    return p


def signatures(p: pd.DataFrame) -> pd.DataFrame:
    q = p.dropna(subset=["index"]).copy()
    share_like = q.kind.isin(["share", "flag", "cat"])
    keep = ~share_like | (q[["segment_value", "overall_value"]].max(axis=1) >= MIN_PREVALENCE)
    keep &= ~q.kind.eq("cat") | (q.segment_value >= MIN_LEVEL_SHARE)
    q = q[keep & (q["index"] > 0)]
    q["strength"] = np.abs(np.log(q["index"] / 100))
    return q.sort_values("strength", ascending=False).groupby("segment_id").head(SIGNATURE_SIZE)


def fmt_value(row) -> str:
    v, o = row.segment_value, row.overall_value
    if row.kind in ("share", "flag", "cat"):
        return f"{v:.0%} vs {o:.0%}"
    if row.kind == "price" and row.feature != "price_level_proxy":
        return f"£{v:,.0f} vs £{o:,.0f}"
    return f"{v:,.2f} vs {o:,.2f}"


def plot_heatmap(p: pd.DataFrame, labels: dict, path) -> None:
    h = p[p.feature.isin(HEATMAP_FEATURES) & p.level.eq("")].pivot(index="feature", columns="segment_id", values="index")
    h = h.reindex(HEATMAP_FEATURES)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(np.log2(h.clip(12.5, 800) / 100), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_yticks(range(len(h)), h.index)
    ax.set_xticks(range(h.shape[1]), [labels.get(s, f"Segment {s}") for s in h.columns], rotation=30, ha="right")
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            ax.text(j, i, f"{h.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="log2(index / 100); 0 = average")
    ax.set_title(f"Segment profiles, index vs average customer (100 = average), {MODEL_VERSION}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_profile_table(p: pd.DataFrame) -> None:
    out = p.assign(model_version=MODEL_VERSION)[["model_version", "segment_id", "feature", "level", "group", "kind",
                                                 "model_input", "statistic", "segment_value", "overall_value", "index"]]
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS ds.segment_profile (
                           model_version text, segment_id int, feature text, level text, feature_group text,
                           kind text, model_input boolean, statistic text, segment_value double precision,
                           overall_value double precision, index_vs_avg double precision,
                           PRIMARY KEY (model_version, segment_id, feature, level))""")
        cur.execute("DELETE FROM ds.segment_profile WHERE model_version = %s", (MODEL_VERSION,))
        # numeric features have level '' (part of the key): read empty CSV fields there as '', not NULL
        with cur.copy("COPY ds.segment_profile FROM STDIN WITH (FORMAT csv, FORCE_NOT_NULL (level))") as cp:
            cp.write(buf.getvalue())


def apply_names(segment_ids: list[int]) -> None:
    if not NAMES:
        print("  NAMES not filled yet: segment names stay as placeholders")
        return
    if sorted(NAMES) != sorted(segment_ids):
        raise ValueError(f"NAMES covers {sorted(NAMES)}, segments are {sorted(segment_ids)}")
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS ds.segment_names (
                           model_version text, segment_id int, segment_name text NOT NULL, description text NOT NULL,
                           PRIMARY KEY (model_version, segment_id))""")
        cur.execute("DELETE FROM ds.segment_names WHERE model_version = %s", (MODEL_VERSION,))
        cur.executemany("INSERT INTO ds.segment_names VALUES (%s, %s, %s, %s)",
                        [(MODEL_VERSION, s, n, desc) for s, (n, desc) in NAMES.items()])
        cur.execute("""UPDATE ds.customer_segments c SET segment_name = n.segment_name
                       FROM ds.segment_names n
                       WHERE c.model_version = n.model_version AND c.segment_id = n.segment_id
                         AND c.model_version = %s""", (MODEL_VERSION,))
    print(f"  applied {len(NAMES)} segment names")


def write_report(stats, channels, sig, path) -> None:
    lines = [f"# Segment profiles ({MODEL_VERSION})", "",
             "Index = segment value / average customer x 100. Prices use medians; other numerics use means.",
             "Late rates are per payment already due; monthly rent is shown separately because renters pay often.",
             "Names were written from this evidence, not from the answer key.", ""]
    for _, s in stats.iterrows():
        sid = int(s.segment_id)
        name, desc = NAMES.get(sid, (f"Segment {sid}", ""))
        top_ch = channels[channels.segment_id == sid].nlargest(3, "lead_share")
        budget = (f"median budget £{s.median_budget_gbp:,.0f}" if pd.notna(s.median_budget_gbp) else "") + \
                 (f"; median rent budget £{s.median_rent_budget_pcm:,.0f} pcm" if pd.notna(s.median_rent_budget_pcm) else "")
        lines += [f"## {sid}. {name}", "", desc, "",
                  f"- **Size:** {s.customers:,.0f} customers ({s.customer_share:.1%}); buyers {s.buyer_share:.0%}; "
                  f"overseas {s.overseas_share:.0%}; {budget}; budget missing {s.budget_missing_share:.0%}",
                  f"- **Pipeline:** {s.leads_per_customer:.2f} leads per customer; conversion "
                  f"{(s.conversion_rate or 0):.1%}; median {s.median_days_to_close or 0:.0f} days to close; "
                  f"median first reply {s.median_first_reply_min or 0:.0f} min",
                  f"- **Value:** {s.deals:,.0f} deals, {s.deal_value_share:.1%} of deal value; median deal "
                  f"£{(s.median_deal_value_gbp or 0):,.0f}; lettings {(s.lettings_deal_share or 0):.0%} of deals",
                  f"- **Payments:** late rate {(s.late_rate_all or 0):.1%} overall; monthly rent "
                  + (f"{s.late_rate_monthly_rent:.1%}" if pd.notna(s.late_rate_monthly_rent) else "n/a")
                  + f"; other payments {(s.late_rate_other or 0):.1%}; overdue £{s.overdue_gbp:,.0f}",
                  f"- **Top channels:** " + ", ".join(f"{r.channel} {r.lead_share:.0%}" for r in top_ch.itertuples()),
                  f"- **Reach:** {s.consenting_customers:,.0f} consenting customers; average assignment confidence "
                  f"{s.avg_confidence:.2f}", "", "| Distinctive trait | Segment vs average | Index |", "|---|---|---|"]
        for r in sig[sig.segment_id == sid].itertuples():
            label = f"{r.feature} = {r.level}" if r.level else r.feature
            lines.append(f"| {label} | {fmt_value(r)} | {r.index:.0f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"Segment statistics for {MODEL_VERSION} (sql/ds/ds05_segment_stats.sql)")
    run_sql_file("ds/ds05_segment_stats.sql", params={"MODEL_VERSION": MODEL_VERSION})
    stats = query_df("SELECT * FROM ds.segment_stats WHERE model_version = %s ORDER BY segment_id", (MODEL_VERSION,))
    channels = query_df("SELECT * FROM ds.segment_channels WHERE model_version = %s", (MODEL_VERSION,))

    d = load_profile_frame()
    prof = build_profile(d)
    sig = signatures(prof)
    write_profile_table(prof)
    apply_names(stats.segment_id.astype(int).tolist())

    labels = {s: f"{s}. {NAMES[s][0]}" if s in NAMES else f"Segment {s}" for s in stats.segment_id.astype(int)}
    DS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(DS_REPORTS_DIR / "d05_segment_stats.csv", index=False)
    write_report(stats, channels, sig, DS_REPORTS_DIR / "d05_segment_profiles.md")
    plot_heatmap(prof, labels, DS_REPORTS_DIR / "d05_index_heatmap.png")

    tracking.setup()
    with mlflow.start_run(run_name="d05-profiles"):
        mlflow.set_tags({"step": "d05", "model_version": MODEL_VERSION, "named": str(bool(NAMES))})
        for name in ("d05_segment_stats.csv", "d05_segment_profiles.md", "d05_index_heatmap.png"):
            mlflow.log_artifact(str(DS_REPORTS_DIR / name))

    cols = ["segment_id", "customer_share", "buyer_share", "overseas_share", "conversion_rate", "deal_value_share",
            "late_rate_all", "late_rate_monthly_rent", "late_rate_other"]
    print(stats[cols].round(3).to_string(index=False))
    for sid in stats.segment_id.astype(int):
        s = sig[sig.segment_id == sid]
        traits = "; ".join(f"{r.feature}{'=' + r.level if r.level else ''} {fmt_value(r)} ({r.index:.0f})"
                           for r in s.itertuples())
        print(f"\n[{labels[sid]}] {traits}")


if __name__ == "__main__":
    main()
