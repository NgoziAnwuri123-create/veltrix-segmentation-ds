-- ds04_customer_segments.sql
-- Final segment assignment consumed by the dashboard, CRM exports and monitoring, plus the outcome tests
-- that justified the choice. One row per customer per model version; d04 replaces its own version's rows.

CREATE SCHEMA IF NOT EXISTS ds;

CREATE TABLE IF NOT EXISTS ds.customer_segments (
    customer_id     text NOT NULL,
    model_version   text NOT NULL,              -- e.g. seg-v1.0
    segment_id      int  NOT NULL,              -- 1 = largest segment
    segment_name    text NOT NULL,              -- placeholder until d05 names the segments
    confidence      double precision NOT NULL,  -- 0 = on a segment boundary, 1 = clearly inside
    candidate_id    text NOT NULL,
    assigned_at     timestamp NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, model_version)
);
CREATE INDEX IF NOT EXISTS ix_customer_segments_version_segment ON ds.customer_segments (model_version, segment_id);

CREATE TABLE IF NOT EXISTS ds.segment_validation (
    model_version  text NOT NULL,
    candidate_id   text NOT NULL,
    outcome        text NOT NULL,
    test           text NOT NULL,
    population     text NOT NULL,
    n              bigint NOT NULL,
    statistic      double precision,
    p_value        double precision,
    effect_size    double precision,
    effect_measure text,
    PRIMARY KEY (model_version, candidate_id, outcome)
);
