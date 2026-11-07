-- ds05_segment_stats.sql
-- Segment statistics for one model version ({{MODEL_VERSION}}), computed in SQL so the dashboard can
-- reconcile every number. KPI definitions as in ds02: conversion = won / closed leads; deal value = agreed
-- price (sales) or rent x tenancy months (lettings); late rate = (paid late + overdue) / payments already due.
--   ds.segment_stats     one row per segment
--   ds.segment_channels  enquiries by channel per segment
-- Late payment is reported per payment and split by monthly rent vs other payments, because renters make many
-- small payments and would otherwise look worse simply for paying more often.

CREATE SCHEMA IF NOT EXISTS ds;

CREATE TABLE IF NOT EXISTS ds.segment_stats (
    model_version            text NOT NULL,
    segment_id               int  NOT NULL,
    customers                bigint,
    customer_share           double precision,
    buyer_share              double precision,
    overseas_share           double precision,
    median_budget_gbp        double precision,   -- buyers who stated one
    median_rent_budget_pcm   double precision,   -- renters who stated one
    budget_missing_share     double precision,
    leads                    bigint,
    leads_per_customer       double precision,
    closed_leads             bigint,
    won_leads                bigint,
    conversion_rate          double precision,
    median_days_to_close     double precision,
    median_first_reply_min   double precision,
    deals                    bigint,
    deal_value_gbp           double precision,
    deal_value_share         double precision,
    median_deal_value_gbp    double precision,
    lettings_deal_share      double precision,
    payments_due             bigint,
    late_rate_all            double precision,
    late_rate_monthly_rent   double precision,
    late_rate_other          double precision,
    overdue_gbp              double precision,
    consenting_customers     bigint,
    avg_confidence           double precision,
    PRIMARY KEY (model_version, segment_id)
);

CREATE TABLE IF NOT EXISTS ds.segment_channels (
    model_version  text NOT NULL,
    segment_id     int  NOT NULL,
    channel        text NOT NULL,
    channel_group  text NOT NULL,
    leads          bigint NOT NULL,
    lead_share     double precision NOT NULL,   -- share of this segment's leads
    won_leads      bigint NOT NULL,
    PRIMARY KEY (model_version, segment_id, channel)
);

DELETE FROM ds.segment_stats    WHERE model_version = {{MODEL_VERSION}};
DELETE FROM ds.segment_channels WHERE model_version = {{MODEL_VERSION}};

INSERT INTO ds.segment_stats
WITH seg AS (
    SELECT customer_id, segment_id, confidence FROM ds.customer_segments WHERE model_version = {{MODEL_VERSION}}
),
cust AS (
    SELECT s.segment_id,
           count(*)                                                              AS customers,
           avg((c.primary_intent = 'Buy')::int)                                  AS buyer_share,
           avg(c.is_overseas::int)                                               AS overseas_share,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY c.budget_gbp)
               FILTER (WHERE c.primary_intent = 'Buy')                           AS median_budget_gbp,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY c.rent_budget_pcm_gbp)
               FILTER (WHERE c.primary_intent = 'Rent')                          AS median_rent_budget_pcm,
           avg(coalesce(c.budget_missing, c.rent_budget_missing)::int)           AS budget_missing_share,
           count(*) FILTER (WHERE c.consent_marketing)                           AS consenting_customers,
           avg(s.confidence)                                                     AS avg_confidence
    FROM seg s JOIN clean.customers c USING (customer_id)
    GROUP BY s.segment_id
),
lead AS (
    SELECT s.segment_id,
           count(*)                                       AS leads,
           count(*) FILTER (WHERE l.is_closed)            AS closed_leads,
           count(*) FILTER (WHERE l.is_won)               AS won_leads,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY l.days_to_close) AS median_days_to_close
    FROM seg s JOIN clean.leads l USING (customer_id)
    GROUP BY s.segment_id
),
reply AS (
    SELECT s.segment_id, percentile_cont(0.5) WITHIN GROUP (ORDER BY o.med_first_reply_min) AS median_first_reply_min
    FROM seg s JOIN ds.customer_outcomes o USING (customer_id)
    GROUP BY s.segment_id
),
deal AS (
    SELECT s.segment_id,
           count(*)                                                   AS deals,
           sum(t.deal_value_gbp)                                      AS deal_value_gbp,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY t.deal_value_gbp) AS median_deal_value_gbp,
           avg((NOT t.is_sale)::int)                                  AS lettings_deal_share
    FROM seg s JOIN clean.transactions t USING (customer_id)
    GROUP BY s.segment_id
),
pay AS (
    SELECT s.segment_id,
           count(*) FILTER (WHERE p.is_due)                                                         AS payments_due,
           avg(p.is_late::int) FILTER (WHERE p.is_due)                                              AS late_rate_all,
           avg(p.is_late::int) FILTER (WHERE p.is_due AND p.payment_type = 'Monthly rent')          AS late_rate_monthly_rent,
           avg(p.is_late::int) FILTER (WHERE p.is_due AND p.payment_type <> 'Monthly rent')         AS late_rate_other,
           coalesce(sum(p.amount_gbp) FILTER (WHERE p.status = 'Overdue'), 0)                       AS overdue_gbp
    FROM seg s
    JOIN clean.transactions t USING (customer_id)
    JOIN clean.payments p USING (transaction_id)
    GROUP BY s.segment_id
)
SELECT {{MODEL_VERSION}}, c.segment_id, c.customers, c.customers::double precision / sum(c.customers) OVER (),
       c.buyer_share, c.overseas_share, c.median_budget_gbp, c.median_rent_budget_pcm, c.budget_missing_share,
       coalesce(l.leads, 0), coalesce(l.leads, 0)::double precision / c.customers,
       coalesce(l.closed_leads, 0), coalesce(l.won_leads, 0),
       l.won_leads::double precision / nullif(l.closed_leads, 0), l.median_days_to_close, r.median_first_reply_min,
       coalesce(d.deals, 0), coalesce(d.deal_value_gbp, 0),
       coalesce(d.deal_value_gbp, 0) / sum(coalesce(d.deal_value_gbp, 0)) OVER (),
       d.median_deal_value_gbp, d.lettings_deal_share,
       coalesce(p.payments_due, 0), p.late_rate_all, p.late_rate_monthly_rent, p.late_rate_other,
       coalesce(p.overdue_gbp, 0), c.consenting_customers, c.avg_confidence
FROM cust c
LEFT JOIN lead  l USING (segment_id)
LEFT JOIN reply r USING (segment_id)
LEFT JOIN deal  d USING (segment_id)
LEFT JOIN pay   p USING (segment_id);

INSERT INTO ds.segment_channels
SELECT {{MODEL_VERSION}}, s.segment_id, l.channel, l.channel_group,
       count(*), count(*)::double precision / sum(count(*)) OVER (PARTITION BY s.segment_id),
       count(*) FILTER (WHERE l.is_won)
FROM ds.customer_segments s
JOIN clean.leads l USING (customer_id)
WHERE s.model_version = {{MODEL_VERSION}}
GROUP BY s.segment_id, l.channel, l.channel_group;
