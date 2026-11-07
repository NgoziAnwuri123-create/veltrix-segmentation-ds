-- ds07_monitoring.sql
-- Monthly scoring and drift monitoring for the champion segmentation model (d07).
--   ds.segment_score_history  every customer's segment at each refresh (segment movement over time)
--   ds.monitoring_drift       one row per check per refresh, with threshold and status (ok / watch / alert)
--   ds.monitoring_runs        one row per refresh: counts, retrain recommendation and reasons
--   ds.v_segment_size_trend   segment shares per refresh vs the published reference (dashboard / alerts)

CREATE SCHEMA IF NOT EXISTS ds;

CREATE TABLE IF NOT EXISTS ds.segment_score_history (
    as_of          date NOT NULL,
    model_version  text NOT NULL,
    customer_id    text NOT NULL,
    segment_id     int  NOT NULL,
    confidence     double precision NOT NULL,
    PRIMARY KEY (as_of, model_version, customer_id)
);

CREATE TABLE IF NOT EXISTS ds.monitoring_drift (
    as_of          date NOT NULL,
    model_version  text NOT NULL,
    metric         text NOT NULL,     -- segment_size_change / feature_psi / migration / low_confidence_change
    item           text NOT NULL,     -- segment, feature or 'all'
    value          double precision,
    reference      double precision,
    threshold      double precision,
    status         text NOT NULL CHECK (status IN ('ok', 'watch', 'alert')),
    PRIMARY KEY (as_of, model_version, metric, item)
);

CREATE TABLE IF NOT EXISTS ds.monitoring_runs (
    as_of                date NOT NULL,
    model_version        text NOT NULL,
    customers_scored     bigint NOT NULL,
    new_customers        bigint NOT NULL,
    segment_changes      bigint NOT NULL,   -- customers whose current segment changed at this refresh
    alerts               int NOT NULL,
    watches              int NOT NULL,
    retrain_recommended  boolean NOT NULL,
    reasons              text,
    mlflow_run_id        text,
    run_at               timestamp NOT NULL DEFAULT now(),
    PRIMARY KEY (as_of, model_version)
);

CREATE OR REPLACE VIEW ds.v_segment_size_trend AS
SELECT h.as_of, h.model_version, h.segment_id, n.segment_name,
       count(*)                                                                  AS customers,
       count(*)::double precision / sum(count(*)) OVER (PARTITION BY h.as_of, h.model_version) AS share,
       s.customer_share                                                          AS reference_share,
       count(*)::double precision / sum(count(*)) OVER (PARTITION BY h.as_of, h.model_version)
           / s.customer_share - 1                                                AS relative_change
FROM ds.segment_score_history h
JOIN ds.segment_stats s ON s.model_version = h.model_version AND s.segment_id = h.segment_id
LEFT JOIN ds.segment_names n ON n.model_version = h.model_version AND n.segment_id = h.segment_id
GROUP BY h.as_of, h.model_version, h.segment_id, n.segment_name, s.customer_share;
