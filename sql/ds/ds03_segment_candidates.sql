-- ds03_segment_candidates.sql
-- Candidate segmentations from d02/d03, compared in d04 and shared with DA as placeholder labels.
-- Created once; each step replaces only its own candidate_id rows.

CREATE SCHEMA IF NOT EXISTS ds;

CREATE TABLE IF NOT EXISTS ds.candidate_registry (
    candidate_id   text PRIMARY KEY,          -- e.g. baseline-kmeans-k8
    model          text NOT NULL,
    k              int  NOT NULL,
    feature_space  text NOT NULL,
    mlflow_run_id  text NOT NULL,
    metrics        jsonb NOT NULL,
    created_at     timestamp NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ds.segment_candidates (
    candidate_id  text NOT NULL REFERENCES ds.candidate_registry (candidate_id) ON DELETE CASCADE,
    customer_id   text NOT NULL,
    segment       int  NOT NULL,
    PRIMARY KEY (candidate_id, customer_id)
);
