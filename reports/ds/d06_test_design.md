# Campaign test design: seg-test-01 (seg-v1.0)

**Question: does segment-tailored marketing beat one-size-fits-all?**

## Design

- **Hypothesis:** customers who receive their segment's tailored campaign enquire more often than customers who receive the generic campaign.
- **Primary KPI:** share of the cohort making >= 1 new lead within 84 days of launch (12-week enquiry rate). Conversion takes 30-60 days and cannot be read in time.
- **Window:** 2026-11-16 to 2027-02-07 (12 weeks).
- **Cohort:** 42,678 consented customers in segments 2-8 with no open lead on 2026-10-01; frozen (customers who join later are not in the test). Window shoppers are excluded: they almost never enquire and need an engagement-KPI nudge test of their own.
- **Arms:** treatment 21,244, control 21,434, split 50/50 within each segment by hash (salt `veltrix-seg-test-01`).
- **Treatment:** the segment's tailored brief (below). **Control:** Standard Veltrix newsletter and ads: new listings and market updates, same channels and send volume as the tailored arm, no segment-specific content or offers.
- **Baseline:** 2.79% 12-week enquiry rate in the same window a year earlier (2025-11-17).
- **Detectable effect:** 17% relative lift (2.79% -> 3.26%) at 80% power, alpha 0.05 two-sided.
- **Decision rule:** read once at week 12: tailoring wins if the pooled lift is positive with p < 0.05 and guardrails hold. Weekly numbers are for monitoring only (no early stopping). Per-segment lifts are exploratory: each segment alone is underpowered.
- **Guardrails:** hide-ad + unsubscribe share in treatment no more than 1 point above control; observed treatment share within 1 point of 50% (sample-ratio mismatch).
- **Secondary (read at +90 days):** viewings booked, conversion of test-window leads, deal value.
- **Monitoring:** `ds.v_campaign_test_weekly` (per week, per segment and all segments).

## Tailored briefs

- **2:** School-catchment and chain-management content; semi-detached South East listings; phone follow-up.
- **3:** LISA and shared-ownership explainers and Midlands/North flat listings via TikTok/Instagram, mobile-first.
- **4:** New lettings alerts by WhatsApp; fast-response viewing slots; flats around the stated rent.
- **5:** High-yield Northern stock and SPV-friendly buying guides via email and referral partners.
- **6:** London new-build and off-plan launches via WeChat and overseas agents; virtual tours; payment-plan clarity.
- **7:** Chain-free bungalow listings in the South West and Wales; phone and branch appointments.
- **8:** Relocation-partner bundles for company lets; fast viewing and referencing.

## Sizing (segment 0 = the pooled primary analysis)

| Segment | Name | Audience | Baseline rate | MDE (50/50) |
|---|---|---|---|---|
| 0 | All test segments (primary) | 42,678 | 2.79% | 17% |
| 2 | Family home movers | 10,868 | 1.49% | 49% |
| 3 | First-time buyers | 8,513 | 3.56% | 34% |
| 4 | Young renters | 9,552 | 2.42% | 40% |
| 5 | Buy-to-let investors | 6,263 | 5.43% | 32% |
| 6 | Overseas investors | 3,661 | 1.38% | 95% |
| 7 | Downsizers | 2,924 | 2.14% | 83% |
| 8 | Corporate relocations | 897 | 5.19% | 96% |

## Alternatives considered

- One segment, 6 weeks, enquiry KPI (the original plan): MDE 43-140% for every segment. Rejected: existing customers rarely re-enquire, so the test could not detect a realistic 10-20% lift.
- Buy-to-let only, 6 weeks, engagement KPI (logged-in session): MDE 19%, but engagement is a weaker proxy for revenue.
- Tailored vs generic, 6 weeks: MDE 22%, just over the 20% bar.

## Randomisation checks

| Feature | SMD all | Max abs SMD within a segment |
|---|---|---|
| n_leads | +0.005 | 0.046 |
| n_sessions | +0.003 | 0.050 |
| n_campaign_responses | +0.004 | 0.029 |
| dv_mobile | +0.003 | 0.043 |
| share_night_sessions | +0.007 | 0.042 |
| budget_missing | +0.001 | 0.027 |
| tenure_days | +0.004 | 0.037 |
| ch_portal | +0.008 | 0.026 |

A/A backtest on 2025-11-17 (no test ran): treatment 2.73% vs control 2.85%, lift -4.3%, z = -0.67, p = 0.51. Monitoring view returned 12 weeks and matches a direct computation: True.
