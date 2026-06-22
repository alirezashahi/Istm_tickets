"""Stage 6 — evaluation.

Reports per-stage and end-to-end metrics.

Key design:
  - Subcategory models are evaluated on tickets routed by their TRUE parent
    category (isolates subcat quality from upstream errors).
  - End-to-end numbers use the predicted category (shows real-world accuracy).
  - Primary metric is macro-F1; accuracy and weighted-F1 are always reported
    alongside, never alone.

Usage:
    python evaluate.py
"""
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from clean import clean_text, load_and_clean
from scipy.sparse import hstack

from config import (
    CATEGORY_MODEL_CALIBRATED_PATH,
    CATEGORY_MODEL_PATH,
    CATEGORY_TRANSFORMERS_PATH,
    RANDOM_SEED,
    SERVICE_MODEL_PATH,
    SERVICE_TRANSFORMERS_PATH,
    SUBCAT_MODEL_DIR,
    TARGET_CATEGORY,
    TARGET_SERVICE,
    TARGET_SUBCAT,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}\nRun the corresponding training script first.")
    return joblib.load(path)


def _metrics(y_true, y_pred, label: str) -> dict:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    log.info(
        "%s  →  Macro-F1: %.4f  Weighted-F1: %.4f  Accuracy: %.4f",
        label, macro_f1, weighted_f1, acc,
    )
    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "accuracy": acc}


def _safe_name(category: str) -> str:
    import re
    return re.sub(r"[^\w\-]", "_", category)


def _build_svc_X(df: pd.DataFrame, transformers: dict):
    return hstack([
        transformers["tfidf_name"].transform(df["ProfileFullName"].fillna("").values),
        transformers["tfidf_subject"].transform(df["Subject"].map(clean_text).values),
        transformers["tfidf_symptom"].transform(df["Symptom"].map(clean_text).values),
    ])


def _build_cat_X(df: pd.DataFrame, transformers: dict):
    name_col = df["ProfileFullName"].fillna("").values
    subj_col = df["Subject"].map(clean_text).values
    symp_col = df["Symptom"].map(clean_text).values
    return hstack([
        transformers["tfidf_name"].transform(name_col),
        transformers["tfidf_subject"].transform(subj_col),
        transformers["tfidf_symptom"].transform(symp_col),
    ])


# ── Stage evaluations ─────────────────────────────────────────────────────────

def evaluate_service(df_test: pd.DataFrame) -> dict:
    log.info("\n=== Service Stage ===")
    svc_model = _load_model(SERVICE_MODEL_PATH)
    svc_transformers = _load_model(SERVICE_TRANSFORMERS_PATH)

    X = _build_svc_X(df_test, svc_transformers)
    y_true = df_test[TARGET_SERVICE].values
    y_pred = svc_model.predict(X)

    metrics = _metrics(y_true, y_pred, "Service")
    log.info("\n%s", classification_report(y_true, y_pred, zero_division=0))
    return {"metrics": metrics, "y_pred": y_pred}


def evaluate_category(df_test: pd.DataFrame, service_pred: np.ndarray | None = None) -> dict:
    log.info("\n=== Category Stage ===")
    cat_model = _load_model(CATEGORY_MODEL_PATH)
    cat_transformers = _load_model(CATEGORY_TRANSFORMERS_PATH)

    X = _build_cat_X(df_test, cat_transformers)
    y_true = df_test[TARGET_CATEGORY].values
    y_pred = cat_model.predict(X)

    overall = _metrics(y_true, y_pred, "Category")
    log.info("\n%s", classification_report(y_true, y_pred, zero_division=0))
    return {"metrics": overall, "y_pred": y_pred}


def evaluate_subcategory(
    df_test: pd.DataFrame,
    use_true_category: bool = True,
    category_pred: np.ndarray | None = None,
) -> dict:
    mode = "true-category-routed" if use_true_category else "predicted-category-routed"
    log.info("\n=== Subcategory Stage (%s) ===", mode)

    index_path = SUBCAT_MODEL_DIR / "_index.joblib"
    if not index_path.exists():
        raise FileNotFoundError(f"Subcategory index not found: {index_path}")
    index = joblib.load(index_path)

    flat_cats: set[str] = set(index.get("flat_categories", []))
    trained_cats: list[str] = index.get("trained_categories", [])

    # Routing column
    if use_true_category:
        route_col = TARGET_CATEGORY
    else:
        # inject predicted categories into a temp column
        df_test = df_test.copy()
        df_test["_pred_category"] = category_pred
        route_col = "_pred_category"

    per_cat_rows: list[dict] = []
    all_y_true: list[str] = []
    all_y_pred: list[str] = []

    for cat in trained_cats:
        model_path = SUBCAT_MODEL_DIR / f"{_safe_name(cat)}.joblib"
        if not model_path.exists():
            log.warning("  Missing model file: %s", model_path)
            continue

        art = joblib.load(model_path)
        calibrated = art["calibrated"]
        excluded = art.get("excluded_subcats", [])
        classes = art["classes"]

        # Route test rows by the selected category column
        mask = df_test[route_col] == cat
        sub = df_test[mask].copy()
        if len(sub) == 0:
            continue

        # Exclude rows whose true subcategory is below the support floor
        sub = sub[~sub[TARGET_SUBCAT].isin(excluded)]
        if len(sub) == 0:
            continue

        X = _build_svc_X(sub, art["transformers"])
        proba = calibrated.predict_proba(X)
        y_pred = np.array(classes)[proba.argmax(axis=1)]

        macro_f1 = f1_score(sub[TARGET_SUBCAT].values, y_pred, average="macro", zero_division=0)
        support = len(sub)
        log.info("  %-35s  support=%4d  macro-F1=%.4f", cat, support, macro_f1)

        all_y_true.extend(sub[TARGET_SUBCAT].values)
        all_y_pred.extend(y_pred)

        per_cat_rows.append({
            "category": cat,
            "support": support,
            "macro_f1": round(macro_f1, 4),
        })

    # Per-category table
    per_cat_df = pd.DataFrame(per_cat_rows).sort_values("macro_f1", ascending=False)
    log.info("\nPer-category summary:\n%s", per_cat_df.to_string(index=False))

    overall = _metrics(np.array(all_y_true), np.array(all_y_pred), f"Subcategory ({mode})")

    return {
        "metrics": overall,
        "per_category": per_cat_df,
    }


def evaluate_end_to_end(df_test: pd.DataFrame) -> dict:
    """Full pipeline: predicted service → predicted category → predicted subcategory."""
    log.info("\n=== End-to-End Evaluation (predicted routing) ===")
    log.info("NOTE: End-to-end accuracy will be lower than stage-isolated numbers.")
    log.info("      This is expected after removing Altro (18%% of data). Judge by macro-F1.\n")

    # Service predictions
    svc_model = _load_model(SERVICE_MODEL_PATH)
    svc_transformers = _load_model(SERVICE_TRANSFORMERS_PATH)
    X_svc = _build_svc_X(df_test, svc_transformers)
    svc_pred = svc_model.predict(X_svc)

    # Category predictions — global model, no service routing needed
    cat_model = _load_model(CATEGORY_MODEL_PATH)
    cat_transformers = _load_model(CATEGORY_TRANSFORMERS_PATH)
    X_cat = _build_cat_X(df_test, cat_transformers)
    cat_pred = cat_model.predict(X_cat)

    # Subcategory (using predicted category)
    result = evaluate_subcategory(df_test, use_true_category=False, category_pred=cat_pred)

    log.info("\nEnd-to-end subcategory:")
    log.info("  Macro-F1:     %.4f", result["metrics"]["macro_f1"])
    log.info("  Weighted-F1:  %.4f", result["metrics"]["weighted_f1"])
    log.info("  Accuracy:     %.4f", result["metrics"]["accuracy"])
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Reproduce the exact same splits used during training
    _noise = r"(?i)(cms|altro|other|z-other)"

    df_svc, _ = load_and_clean("service")
    _svc_mask = (
        df_svc[TARGET_CATEGORY].str.contains(_noise, regex=True, na=False) |
        df_svc[TARGET_SUBCAT].str.contains(_noise, regex=True, na=False)
    )
    df_svc = df_svc[~_svc_mask].copy()
    _, df_test_svc = train_test_split(
        df_svc, test_size=TEST_SIZE, stratify=df_svc[TARGET_SERVICE].values, random_state=RANDOM_SEED
    )

    df_cat, _ = load_and_clean("category")
    # Reproduce the same noise filter used in train_category.py
    _noise = r"(?i)(cms|altro|other|z-other)"
    _mask = (
        df_cat[TARGET_CATEGORY].str.contains(_noise, regex=True, na=False) |
        df_cat[TARGET_SUBCAT].str.contains(_noise, regex=True, na=False)
    )
    df_cat = df_cat[~_mask].copy()
    # Mirror train_category.py: drop categories too rare to stratify
    _min_count = max(2, int(np.ceil(1 / TEST_SIZE)))
    _counts = df_cat[TARGET_CATEGORY].value_counts()
    _too_few = _counts[_counts < _min_count].index
    if len(_too_few):
        df_cat = df_cat[~df_cat[TARGET_CATEGORY].isin(_too_few)].copy()
    _, df_test_cat = train_test_split(
        df_cat, test_size=TEST_SIZE, stratify=df_cat[TARGET_CATEGORY].values, random_state=RANDOM_SEED
    )

    df_sub, _ = load_and_clean("subcategory")
    _, df_test_sub = train_test_split(
        df_sub, test_size=TEST_SIZE, stratify=df_sub[TARGET_CATEGORY].values, random_state=RANDOM_SEED
    )

    evaluate_service(df_test_svc)
    evaluate_category(df_test_cat)
    evaluate_subcategory(df_test_sub, use_true_category=True)
    evaluate_end_to_end(df_test_sub)


if __name__ == "__main__":
    main()
