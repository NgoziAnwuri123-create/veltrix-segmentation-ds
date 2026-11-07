"""DS d06: design the segment campaign test (tailored vs generic).

Question: does segment-tailored marketing beat one-size-fits-all?
Design (chosen with the project lead after sizing; see the report for the alternatives):
    cohort     consented customers in TEST_SEGMENTS with no open lead, frozen on the latest data
    arms       50/50 within each segment. Treatment gets its segment's campaign brief; control gets the generic
               campaign through the same channels at the same volume, so only the tailoring differs.
    KPI        12-week enquiry rate: share making >= 1 new lead within TEST_DAYS of launch. (Conversion takes
               30-60 days and cannot be read in time; enquiries sit directly upstream of deals.)
    analysis   pooled two-proportion z-test at the end (no early stopping); per-segment reads are exploratory.
Why not one segment for 6 weeks: existing customers rarely re-enquire (1-3% per 6 weeks), so every single-segment
6-week test had a minimum detectable lift of 43-140%. Window shoppers (segment 1) almost never enquire; they need
an engagement-KPI nudge test of their own.

Sizing uses the same calendar window one year earlier (the test runs through the December slump).
Randomisation: arm = treatment when sha256(salt:customer_id) mod 10,000 < 10,000 x share, within each segment.
Checks: covariate balance (|SMD| < 0.1) overall and per segment, and an A/A backtest on the baseline window
(no campaign then, so the arms must not differ), which also validates the monitoring view.
Writes ds.campaign_tests, ds.campaign_test_assignment; the view ds.v_campaign_test_weekly is DA's monitoring.
"""
import hashlib
from datetime import date, timedelta

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import norm

from config import DATA_TO, DS_REPORTS_DIR, MODEL_VERSION
from src.db import connect, query_df, render_sql, run_sql_file
from src.ds import tracking
from src.ds.candidates import load_features

TEST_ID = "seg-test-01"
TEST_SEGMENTS = [2, 3, 4, 5, 6, 7, 8]
TEST_START = date(2026, 11, 16)          # Monday after design sign-off
TEST_DAYS = 84
TREATMENT_SHARE = 0.5
ALPHA, POWER = 0.05, 0.80
MAX_MDE = 0.20
HASH_SALT = "veltrix-seg-test-01"
BASELINE_START = TEST_START - timedelta(days=364)   # same weekday, one year earlier
AUDIENCE_DATE = DATA_TO + timedelta(days=1)          # cohort frozen on the latest data
BALANCE_FEATURES = ["n_leads", "n_sessions", "n_campaign_responses", "dv_mobile", "share_night_sessions",
                    "budget_missing", "tenure_days", "ch_portal"]
GENERIC_BRIEF = ("Standard Veltrix newsletter and ads: new listings and market updates, same channels and send "
                 "volume as the tailored arm, no segment-specific content or offers.")

# Tailored brief per segment, from the d05 evidence
BRIEFS = {
    2: "School-catchment and chain-management content; semi-detached South East listings; phone follow-up.",
    3: "LISA and shared-ownership explainers and Midlands/North flat listings via TikTok/Instagram, mobile-first.",
    4: "New lettings alerts by WhatsApp; fast-response viewing slots; flats around the stated rent.",
    5: "High-yield Northern stock and SPV-friendly buying guides via email and referral partners.",
    6: "London new-build and off-plan launches via WeChat and overseas agents; virtual tours; payment-plan clarity.",
    7: "Chain-free bungalow listings in the South West and Wales; phone and branch appointments.",
    8: "Relocation-partner bundles for company lets; fast viewing and referencing.",
}


def cohort(start: date) -> pd.DataFrame:
    c = query_df(render_sql("ds/ds06_test_baseline.sql", {"MODEL_VERSION": MODEL_VERSION,
                                                           "WINDOW_START": start.isoformat(),
                                                           "WINDOW_DAYS": TEST_DAYS}))
    return c[c.segment_id.isin(TEST_SEGMENTS)].reset_index(drop=True)


def mde_relative(p: float, n_t: int, n_c: int) -> float:
    """Smallest relative lift r with power POWER for a two-sided two-proportion z-test (unpooled)."""
    if p <= 0 or n_t == 0 or n_c == 0:
        return np.inf
    z = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)
    lo, hi = 0.0, 50.0
    for _ in range(100):
        r = (lo + hi) / 2
        p2 = min(p * (1 + r), 0.999)
        se = np.sqrt(p2 * (1 - p2) / n_t + p * (1 - p) / n_c)
        lo, hi = (lo, r) if (p2 - p) / se >= z else (r, hi)
    return hi


def assign(customer_ids: pd.Series, salt: str, share: float) -> np.ndarray:
    cut = int(round(share * 10_000))
    buckets = customer_ids.map(lambda c: int(hashlib.sha256(f"{salt}:{c}".encode()).hexdigest(), 16) % 10_000)
    return np.where(buckets < cut, "treatment", "control")


def smd(a: pd.Series, b: pd.Series) -> float:
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def two_prop_z(x1, n1, x2, n2) -> tuple[float, float]:
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return float(z), float(2 * norm.sf(abs(z)))


def sizing(base: pd.DataFrame, audience: pd.DataFrame) -> pd.DataFrame:
    names = query_df("SELECT segment_id, segment_name FROM ds.segment_names WHERE model_version = %s",
                     (MODEL_VERSION,)).set_index("segment_id").segment_name
    t = pd.DataFrame({"baseline_rate": base.groupby("segment_id").enquired.mean(),
                      "audience": audience.groupby("segment_id").size()})
    t.loc[0] = [base.enquired.mean(), len(audience)]
    t["segment_name"] = names.reindex(t.index).fillna("All test segments (primary)")
    t["n_treatment"] = (t.audience * TREATMENT_SHARE).round().astype(int)
    t["mde_relative"] = [mde_relative(p, nt, n - nt) for p, nt, n in zip(t.baseline_rate, t.n_treatment, t.audience)]
    return t.sort_index()


def balance(c: pd.DataFrame) -> pd.DataFrame:
    f, _ = load_features()
    fb = c.merge(f[["customer_id"] + BALANCE_FEATURES], on="customer_id")
    rows = []
    for k in BALANCE_FEATURES:
        x = fb[k].astype(float)
        per_seg = [smd(x[(fb.arm == "treatment") & (fb.segment_id == s)], x[(fb.arm == "control") & (fb.segment_id == s)])
                   for s in TEST_SEGMENTS]
        rows.append({"feature": k, "smd_all": smd(x[fb.arm == "treatment"], x[fb.arm == "control"]),
                     "max_abs_smd_within_segment": float(np.max(np.abs(per_seg)))})
    return pd.DataFrame(rows)


def register(conn, test_id, salt, start, baseline_rate, mde, notes, c) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ds.campaign_tests WHERE test_id = %s", (test_id,))
        cur.execute("INSERT INTO ds.campaign_tests (test_id, model_version, segment_ids, description, start_date, "
                    "end_date, treatment_share, primary_kpi, baseline_rate, mde_relative, alpha, power, hash_salt, "
                    "design_notes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (test_id, MODEL_VERSION, TEST_SEGMENTS, "Segment-tailored vs generic campaign", start,
                     start + timedelta(days=TEST_DAYS), TREATMENT_SHARE, f"{TEST_DAYS // 7}-week enquiry rate",
                     baseline_rate, mde, ALPHA, POWER, salt, notes))
        cur.executemany("INSERT INTO ds.campaign_test_assignment VALUES (%s, %s, %s, %s)",
                        [(test_id, r.customer_id, int(r.segment_id), r.arm) for r in c.itertuples()])


def aa_backtest(base: pd.DataFrame) -> dict:
    """The full procedure on the baseline window, when no test ran: arms must not differ, and the monitoring view
    must reproduce a direct computation."""
    aa_id, salt = "aa-backtest", HASH_SALT + "-aa"
    c = base.copy()
    c["arm"] = assign(c.customer_id, salt, TREATMENT_SHARE)
    with connect() as conn:
        register(conn, aa_id, salt, BASELINE_START, 0.0, 0.0, "A/A backtest", c)
    weekly = query_df("SELECT * FROM ds.v_campaign_test_weekly WHERE test_id = %s ORDER BY week, segment_id", (aa_id,))
    with connect() as conn:
        conn.execute("DELETE FROM ds.campaign_tests WHERE test_id = %s", (aa_id,))
    t, h = c[c.arm == "treatment"], c[c.arm == "control"]
    z, p = two_prop_z(t.enquired.sum(), len(t), h.enquired.sum(), len(h))
    final = weekly[(weekly.segment_id == 0) & (weekly.week == weekly.week.max())].iloc[0]
    return {"weeks_in_view": int(weekly.week.max()), "treatment_rate": float(t.enquired.mean()),
            "control_rate": float(h.enquired.mean()), "relative_lift": float(t.enquired.mean() / h.enquired.mean() - 1),
            "z": z, "p_value": p,
            "view_matches_direct": bool(np.isclose(final.treatment_enquiry_rate, t.enquired.mean())
                                        and np.isclose(final.control_enquiry_rate, h.enquired.mean())),
            "weekly": weekly}


def write_report(t, c, bal, aa, path) -> None:
    primary = t.loc[0]
    n_t, n_c = (c.arm == "treatment").sum(), (c.arm == "control").sum()
    seg_rows = [f"| {i} | {r.segment_name} | {r.audience:,.0f} | {r.baseline_rate:.2%} | "
                + ("n/a" if not np.isfinite(r.mde_relative) else f"{r.mde_relative:.0%}") + " |"
                for i, r in t.iterrows()]
    lines = [
        f"# Campaign test design: {TEST_ID} ({MODEL_VERSION})", "",
        "**Question: does segment-tailored marketing beat one-size-fits-all?**", "",
        "## Design", "",
        "- **Hypothesis:** customers who receive their segment's tailored campaign enquire more often than "
        "customers who receive the generic campaign.",
        f"- **Primary KPI:** share of the cohort making >= 1 new lead within {TEST_DAYS} days of launch "
        f"({TEST_DAYS // 7}-week enquiry rate). Conversion takes 30-60 days and cannot be read in time.",
        f"- **Window:** {TEST_START} to {TEST_START + timedelta(days=TEST_DAYS - 1)} ({TEST_DAYS // 7} weeks).",
        f"- **Cohort:** {len(c):,} consented customers in segments {TEST_SEGMENTS[0]}-{TEST_SEGMENTS[-1]} with no "
        f"open lead on {AUDIENCE_DATE}; frozen (customers who join later are not in the test). Window shoppers are "
        "excluded: they almost never enquire and need an engagement-KPI nudge test of their own.",
        f"- **Arms:** treatment {n_t:,}, control {n_c:,}, split 50/50 within each segment by hash (salt "
        f"`{HASH_SALT}`).",
        f"- **Treatment:** the segment's tailored brief (below). **Control:** {GENERIC_BRIEF}",
        f"- **Baseline:** {primary.baseline_rate:.2%} {TEST_DAYS // 7}-week enquiry rate in the same window a year "
        f"earlier ({BASELINE_START}).",
        f"- **Detectable effect:** {primary.mde_relative:.0%} relative lift ({primary.baseline_rate:.2%} -> "
        f"{primary.baseline_rate * (1 + primary.mde_relative):.2%}) at {POWER:.0%} power, alpha {ALPHA} two-sided.",
        f"- **Decision rule:** read once at week {TEST_DAYS // 7}: tailoring wins if the pooled lift is positive with "
        f"p < {ALPHA} and guardrails hold. Weekly numbers are for monitoring only (no early stopping). Per-segment "
        "lifts are exploratory: each segment alone is underpowered.",
        "- **Guardrails:** hide-ad + unsubscribe share in treatment no more than 1 point above control; observed "
        "treatment share within 1 point of 50% (sample-ratio mismatch).",
        "- **Secondary (read at +90 days):** viewings booked, conversion of test-window leads, deal value.",
        "- **Monitoring:** `ds.v_campaign_test_weekly` (per week, per segment and all segments).", "",
        "## Tailored briefs", "", *[f"- **{s}:** {b}" for s, b in BRIEFS.items()], "",
        "## Sizing (segment 0 = the pooled primary analysis)", "",
        "| Segment | Name | Audience | Baseline rate | MDE (50/50) |", "|---|---|---|---|---|", *seg_rows, "",
        "## Alternatives considered", "",
        "- One segment, 6 weeks, enquiry KPI (the original plan): MDE 43-140% for every segment. Rejected: "
        "existing customers rarely re-enquire, so the test could not detect a realistic 10-20% lift.",
        "- Buy-to-let only, 6 weeks, engagement KPI (logged-in session): MDE 19%, but engagement is a weaker proxy "
        "for revenue.",
        "- Tailored vs generic, 6 weeks: MDE 22%, just over the 20% bar.", "",
        "## Randomisation checks", "",
        "| Feature | SMD all | Max abs SMD within a segment |", "|---|---|---|",
        *[f"| {r.feature} | {r.smd_all:+.3f} | {r.max_abs_smd_within_segment:.3f} |" for r in bal.itertuples()], "",
        f"A/A backtest on {BASELINE_START} (no test ran): treatment {aa['treatment_rate']:.2%} vs control "
        f"{aa['control_rate']:.2%}, lift {aa['relative_lift']:+.1%}, z = {aa['z']:.2f}, p = {aa['p_value']:.2f}. "
        f"Monitoring view returned {aa['weeks_in_view']} weeks and matches a direct computation: "
        f"{aa['view_matches_direct']}.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    run_sql_file("ds/ds06_campaign_test.sql")
    print(f"Sizing on {BASELINE_START} (+{TEST_DAYS} days); audience as of {AUDIENCE_DATE}")
    base, c = cohort(BASELINE_START), cohort(AUDIENCE_DATE)
    t = sizing(base, c)
    print(t[["segment_name", "audience", "baseline_rate", "mde_relative"]].round(4).to_string())
    primary = t.loc[0]
    if primary.mde_relative > MAX_MDE:
        raise RuntimeError(f"Pooled MDE {primary.mde_relative:.0%} exceeds {MAX_MDE:.0%}")

    c["arm"] = assign(c.customer_id, HASH_SALT, TREATMENT_SHARE)
    bal = balance(c)
    aa = aa_backtest(base)
    with connect() as conn:
        register(conn, TEST_ID, HASH_SALT, TEST_START, float(primary.baseline_rate), float(primary.mde_relative),
                 "Tailored briefs per segment; control = generic campaign, same channels and volume", c)

    DS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(DS_REPORTS_DIR / "d06_test_sizing.csv")
    bal.to_csv(DS_REPORTS_DIR / "d06_balance.csv", index=False)
    aa["weekly"].to_csv(DS_REPORTS_DIR / "d06_aa_backtest_weekly.csv", index=False)
    write_report(t, c, bal, aa, DS_REPORTS_DIR / "d06_test_design.md")

    tracking.setup()
    with mlflow.start_run(run_name="d06-test-design"):
        mlflow.set_tags({"step": "d06", "model_version": MODEL_VERSION, "test_id": TEST_ID})
        mlflow.log_params({"design": "tailored vs generic", "segments": str(TEST_SEGMENTS), "start": str(TEST_START),
                           "days": TEST_DAYS, "treatment_share": TREATMENT_SHARE, "alpha": ALPHA, "power": POWER,
                           "baseline_window": str(BASELINE_START), "hash_salt": HASH_SALT})
        mlflow.log_metrics({"cohort": len(c), "baseline_rate": float(primary.baseline_rate),
                            "mde_relative": float(primary.mde_relative),
                            "max_abs_smd": float(bal[["smd_all"]].abs().max().iloc[0]),
                            "max_abs_smd_within_segment": float(bal.max_abs_smd_within_segment.max()),
                            "aa_relative_lift": aa["relative_lift"], "aa_p_value": aa["p_value"]})
        for name in ("d06_test_sizing.csv", "d06_balance.csv", "d06_aa_backtest_weekly.csv", "d06_test_design.md"):
            mlflow.log_artifact(str(DS_REPORTS_DIR / name))

    print(f"\nCohort {len(c):,}: treatment {(c.arm == 'treatment').sum():,} / control {(c.arm == 'control').sum():,}")
    print(bal.round(3).to_string(index=False))
    print(f"A/A backtest: treatment {aa['treatment_rate']:.2%} vs control {aa['control_rate']:.2%} "
          f"(lift {aa['relative_lift']:+.1%}, p = {aa['p_value']:.2f}); view weeks {aa['weeks_in_view']}, "
          f"view matches direct: {aa['view_matches_direct']}")


if __name__ == "__main__":
    main()
