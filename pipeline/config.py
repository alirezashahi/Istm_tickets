"""Single source of truth for all constants, paths, and thresholds."""
from pathlib import Path

RANDOM_SEED = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# Accept either the parquet (preferred) or the original CSV
_parquet = DATA_DIR / "raw_tickets.parquet"
_csv = ROOT.parent / "2026-06-05_01.13.20_ItsmIncidentExport.csv"
RAW_DATA_PATH = _parquet if _parquet.exists() else _csv

LABEL_DECISIONS_PATH = DATA_DIR / "label_decisions.csv"

# Model artefact paths
SERVICE_MODEL_PATH = MODEL_DIR / "service_model.joblib"
SERVICE_MODEL_CALIBRATED_PATH = MODEL_DIR / "service_model_calibrated.joblib"
SERVICE_TRANSFORMERS_PATH = MODEL_DIR / "service_transformers.joblib"
CATEGORY_MODEL_PATH = MODEL_DIR / "category_model.joblib"
CATEGORY_MODEL_CALIBRATED_PATH = MODEL_DIR / "category_model_calibrated.joblib"
CATEGORY_TRANSFORMERS_PATH = MODEL_DIR / "category_transformers.joblib"
SUBCAT_MODEL_DIR = MODEL_DIR / "subcategory"
SUBCAT_FULL_MODEL_DIR = MODEL_DIR / "subcategory_full"  # trained without the support floor

# Audit output paths
AUDIT_DIR = DATA_DIR / "audit"

# ── Feature columns ───────────────────────────────────────────────────────────
TEXT_FIELDS = ["Subject", "Symptom"]
SENDER_FIELD = "ProfileFullName"
TARGET_SERVICE = "Service"
TARGET_CATEGORY = "Category"
TARGET_SUBCAT = "Subcategory"

# Columns to drop before any modelling (missing / low signal)
DROP_FIELDS = ["IncidentNumber", "SCMTitle", "CustomerLocation", "Email", "SCMTypeOfRequest"]

# ── Modelling hyperparameters ─────────────────────────────────────────────────
PRIMARY_METRIC = "macro_f1"

# TF-IDF feature budget — matches the known-good 13b baseline (~50k features)
TFIDF_MAX_FEATURES = 50_000

# Minimum (category, subcategory) pair support to include in training
MIN_SUBCAT_SUPPORT = 50

# Stratified train/test split ratio
TEST_SIZE = 0.20

# Top-N senders to keep as explicit categories; rest → "__other__"
SENDER_TOP_N = 200

# label_decisions.csv tag constants — do not change; they match the CSV
TAG_REAL = "REAL"
TAG_TRASHBIN = "TRASHBIN"
TAG_DEFAULT = "DEFAULT"
TAG_FLAT = "FLAT"

ABSTAIN_LABEL = "unspecified (review)"
