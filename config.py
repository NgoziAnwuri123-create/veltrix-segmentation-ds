"""Project-wide paths, database settings and constants."""
from datetime import date
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Paths
RAW_DIR = ROOT / "data" / "raw" / "veltrix_medium"
PROCESSED_DIR = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"
DS_REPORTS_DIR = REPORTS_DIR / "ds"
EXPORTS_DIR = ROOT / "exports"

# MLflow: local SQLite tracking store + file artifacts, both git-ignored
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_ROOT = (ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT = "veltrix-segmentation"

# Database
PG = {
    "user": os.getenv("POSTGRES_USER", "veltrix"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "dbname": os.getenv("POSTGRES_DB", "veltrix"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}
RAW_SCHEMA = "raw"

# Project constants
DATA_FROM = date(2023, 1, 1)
DATA_TO = date(2026, 9, 30)
MODEL_VERSION = "seg-v1.0"
FEATURE_VERSION = "fv1"
RANDOM_SEED = 42

# Source tables: name -> path relative to RAW_DIR (folders hold Parquet part files).
# Ordered so parents load before children. The _restricted answer key is deliberately absent.
TABLES = {
    "locations": "locations.parquet",
    "agents": "agents.parquet",
    "customers": "customers.parquet",
    "campaigns": "campaigns.parquet",
    "properties": "properties.parquet",
    "leads": "leads.parquet",
    "interactions": "interactions.parquet",
    "site_visits": "site_visits.parquet",
    "transactions": "transactions.parquet",
    "payments": "payments.parquet",
    "campaign_responses": "campaign_responses.parquet",
    "web_sessions": "web_sessions",
    "property_views": "property_views",
}
