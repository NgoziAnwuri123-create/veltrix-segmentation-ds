-- ds06_test_baseline.sql
-- Campaign audience and 6-week enquiry outcome, one row per eligible customer.
-- Eligible at window start: created before it, consented to marketing, and no lead open at that moment
-- (customers already in an active deal are being worked by agents and are not a campaign audience).
-- Outcome: whether they made a new lead inside the window, and how many of those leads closed / were won.
-- Used with a past window to size the test (baseline rate) and with the launch date to freeze the test cohort.
-- Parameters: {{MODEL_VERSION}}, {{WINDOW_START}} (date), {{WINDOW_DAYS}} (int as text).

WITH seg AS (
    SELECT s.customer_id, s.segment_id
    FROM ds.customer_segments s
    JOIN clean.customers c USING (customer_id)
    WHERE s.model_version = {{MODEL_VERSION}}
      AND c.consent_marketing
      AND c.created_at < {{WINDOW_START}}::date
      AND NOT EXISTS (SELECT 1 FROM clean.leads l
                      WHERE l.customer_id = s.customer_id
                        AND l.created_date < {{WINDOW_START}}::date
                        AND (l.closed_date IS NULL OR l.closed_date >= {{WINDOW_START}}::date))
),
window_leads AS (
    SELECT l.customer_id, l.is_closed, l.is_won
    FROM clean.leads l
    WHERE l.created_date >= {{WINDOW_START}}::date
      AND l.created_date <  {{WINDOW_START}}::date + {{WINDOW_DAYS}}::int
)
SELECT s.customer_id,
       s.segment_id,
       count(w.customer_id) > 0                                        AS enquired,
       count(w.customer_id) FILTER (WHERE w.is_closed)                 AS window_leads_closed,
       count(w.customer_id) FILTER (WHERE w.is_won)                    AS window_leads_won
FROM seg s
LEFT JOIN window_leads w USING (customer_id)
GROUP BY s.customer_id, s.segment_id;
