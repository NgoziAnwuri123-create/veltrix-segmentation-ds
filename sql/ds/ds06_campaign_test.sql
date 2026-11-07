-- ds06_campaign_test.sql
-- Campaign test registry, frozen customer assignment, and the weekly monitoring view for the primary KPI.
-- The test cohort is frozen at design time: customers who join later are not in the test.
-- Re-runnable: the test tables are recreated (they hold design outputs only, rebuilt by d06).

CREATE SCHEMA IF NOT EXISTS ds;

DROP VIEW IF EXISTS ds.v_campaign_test_weekly;
DROP TABLE IF EXISTS ds.campaign_test_assignment;
DROP TABLE IF EXISTS ds.campaign_tests;

CREATE TABLE ds.campaign_tests (
    test_id          text PRIMARY KEY,
    model_version    text NOT NULL,
    segment_ids      int[] NOT NULL,
    description      text NOT NULL,
    start_date       date NOT NULL,
    end_date         date NOT NULL,              -- exclusive
    treatment_share  double precision NOT NULL,
    primary_kpi      text NOT NULL,
    baseline_rate    double precision NOT NULL,
    mde_relative     double precision NOT NULL,
    alpha            double precision NOT NULL,
    power            double precision NOT NULL,
    hash_salt        text NOT NULL,
    design_notes     text,
    created_at       timestamp NOT NULL DEFAULT now()
);

CREATE TABLE ds.campaign_test_assignment (
    test_id      text NOT NULL REFERENCES ds.campaign_tests (test_id) ON DELETE CASCADE,
    customer_id  text NOT NULL,
    segment_id   int  NOT NULL,
    arm          text NOT NULL CHECK (arm IN ('treatment', 'control')),
    PRIMARY KEY (test_id, customer_id)
);

-- Cumulative primary KPI per week, arm and segment (segment_id 0 = all segments), the treatment-vs-control lift,
-- the guardrail (share with a hide-ad or unsubscribe) and the observed treatment share (sample-ratio check).
-- A week appears once the data reaches its first day.
CREATE VIEW ds.v_campaign_test_weekly AS
WITH weeks AS (
    SELECT t.test_id, gs.week, t.start_date + 7 * gs.week AS week_end
    FROM ds.campaign_tests t
    CROSS JOIN LATERAL generate_series(1, (t.end_date - t.start_date) / 7) AS gs(week)
    WHERE t.start_date + 7 * (gs.week - 1) <= (SELECT max(created_date)::date FROM clean.leads)
),
first_lead AS (
    SELECT a.test_id, a.customer_id, min(l.created_date) AS at
    FROM ds.campaign_test_assignment a
    JOIN ds.campaign_tests t USING (test_id)
    JOIN clean.leads l ON l.customer_id = a.customer_id
                      AND l.created_date >= t.start_date AND l.created_date < t.end_date
    GROUP BY a.test_id, a.customer_id
),
first_negative AS (
    SELECT a.test_id, a.customer_id, min(r.response_time) AS at
    FROM ds.campaign_test_assignment a
    JOIN ds.campaign_tests t USING (test_id)
    JOIN clean.campaign_responses r ON r.customer_id = a.customer_id
                                   AND r.response_time >= t.start_date AND r.response_time < t.end_date
                                   AND r.response_type IN ('Hide ad', 'Unsubscribe')
    GROUP BY a.test_id, a.customer_id
),
cells AS (
    SELECT w.test_id, w.week, w.week_end, a.segment_id, a.arm,
           count(*)                                         AS customers,
           count(*) FILTER (WHERE fl.at < w.week_end)       AS enquirers,
           count(*) FILTER (WHERE fn.at < w.week_end)       AS negatives
    FROM weeks w
    JOIN ds.campaign_test_assignment a USING (test_id)
    LEFT JOIN first_lead fl     ON fl.test_id = a.test_id AND fl.customer_id = a.customer_id
    LEFT JOIN first_negative fn ON fn.test_id = a.test_id AND fn.customer_id = a.customer_id
    GROUP BY w.test_id, w.week, w.week_end, a.segment_id, a.arm
),
with_total AS (
    SELECT * FROM cells
    UNION ALL
    SELECT test_id, week, week_end, 0, arm, sum(customers), sum(enquirers), sum(negatives)
    FROM cells GROUP BY test_id, week, week_end, arm
)
SELECT c.test_id, c.week, c.week_end, c.segment_id,
       coalesce(n.segment_name, 'All segments')                                              AS segment_name,
       max(c.customers) FILTER (WHERE c.arm = 'treatment')                                   AS treatment_customers,
       max(c.customers) FILTER (WHERE c.arm = 'control')                                     AS control_customers,
       max(c.enquirers::double precision / c.customers) FILTER (WHERE c.arm = 'treatment')   AS treatment_enquiry_rate,
       max(c.enquirers::double precision / c.customers) FILTER (WHERE c.arm = 'control')     AS control_enquiry_rate,
       max(c.enquirers::double precision / c.customers) FILTER (WHERE c.arm = 'treatment')
         / nullif(max(c.enquirers::double precision / c.customers) FILTER (WHERE c.arm = 'control'), 0) - 1
                                                                                             AS relative_lift,
       max(c.negatives::double precision / c.customers) FILTER (WHERE c.arm = 'treatment')   AS treatment_negative_rate,
       max(c.negatives::double precision / c.customers) FILTER (WHERE c.arm = 'control')     AS control_negative_rate,
       max(c.customers) FILTER (WHERE c.arm = 'treatment')::double precision / sum(c.customers)
                                                                                             AS observed_treatment_share,
       t.treatment_share                                                                     AS planned_treatment_share
FROM with_total c
JOIN ds.campaign_tests t USING (test_id)
LEFT JOIN ds.segment_names n ON n.model_version = t.model_version AND n.segment_id = c.segment_id
GROUP BY c.test_id, c.week, c.week_end, c.segment_id, n.segment_name, t.treatment_share;
