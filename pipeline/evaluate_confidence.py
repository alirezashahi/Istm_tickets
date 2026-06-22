"""Confidence analysis: floored models vs full models (no support floor).

For both model sets, reports per predicted row:
  - true subcategory, predicted subcategory, top confidence, correct/wrong
  - per-confidence-bucket accuracy (calibration)
  - per-subcategory: avg confidence when correct vs wrong
  - abstain rate at a hypothetical confidence threshold

Also runs end-to-end using the TRUE category (true-category-routed) so the
subcategory confidence is evaluated in isolation from category errors.

Usage:
    python evaluate_confidence.py [--model floored|full|both]
"""
import argparse
import logging
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from clean import load_and_clean
from config import (
    ABSTAIN_CONFIDENCE,
    ABSTAIN_LABEL,
    RANDOM_SEED,
    SUBCAT_FULL_MODEL_DIR,
    SUBCAT_MODEL_DIR,
    TARGET_CATEGORY,
    TARGET_SERVICE,
    TARGET_SUBCAT,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CONFIDENCE_BINS = [0.0, 0.40, 0.60, 0.80, 0.90, 1.01]
BIN_LABELS = ["<0.40 (abstain)", "0.40–0.60", "0.60–0.80", "0.80–0.90", "≥0.90"]


def _safe_name(cat: str) -> str:
    return re.sub(r"[^\w\-]", "_", cat)


def _load_model_dir(model_dir: Path) -> tuple[dict, list[str]]:
    index_path = model_dir / "_index.joblib"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    index = joblib.load(index_path)
    return index, index.get("trained_categories", [])


def _evaluate_one_model_set(
    df_test: pd.DataFrame,
    model_dir: Path,
    label: str,
) -> pd.DataFrame:
    """
    Evaluate subcategory models in model_dir on df_test (true-category-routed).
    Returns a DataFrame with one row per test ticket.
    """
    index, trained_cats = _load_model_dir(model_dir)
    flat_cats: set[str] = set(index.get("flat_categories", []))

    rows = []
    for cat in trained_cats:
        model_path = model_dir / f"{_safe_name(cat)}.joblib"
        if not model_path.exists():
            continue

        art = joblib.load(model_path)
        model = art["model"]
        fp = art["feature_pipeline"]
        classes = np.array(art["classes"])
        excluded = art.get("excluded_subcats", [])

        # Route by true category
        mask = df_test[TARGET_CATEGORY] == cat
        sub = df_test[mask].copy()
        if len(sub) == 0:
            continue

        # For floored models: exclude rows whose true subcat is below the floor
        sub_eval = sub[~sub[TARGET_SUBCAT].isin(excluded)] if excluded else sub
        if len(sub_eval) == 0:
            continue

        X = fp.transform(sub_eval)
        proba = model.predict_proba(X)
        top_idx = proba.argmax(axis=1)
        top_conf = proba.max(axis=1)
        y_pred = classes[top_idx]

        for i, (_, row) in enumerate(sub_eval.iterrows()):
            true_sub = row[TARGET_SUBCAT]
            pred_sub = y_pred[i] if top_conf[i] >= ABSTAIN_CONFIDENCE else ABSTAIN_LABEL
            rows.append({
                "category": cat,
                "true_subcat": true_sub,
                "pred_subcat": pred_sub,
                "confidence": round(float(top_conf[i]), 4),
                "abstained": pred_sub == ABSTAIN_LABEL,
                "correct": pred_sub == true_sub,
                "in_model": true_sub in classes,
            })

    return pd.DataFrame(rows)


def _calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per confidence bin (only non-abstained rows)."""
    non_abstained = df[~df["abstained"]].copy()
    non_abstained["bin"] = pd.cut(
        non_abstained["confidence"],
        bins=CONFIDENCE_BINS,
        labels=BIN_LABELS,
        right=False,
    )
    tbl = (
        non_abstained.groupby("bin", observed=True)
        .agg(
            n=("correct", "count"),
            accuracy=("correct", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .round(3)
    )
    return tbl


def _per_subcat_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per subcategory: support, avg confidence (correct/wrong), coverage, accuracy."""
    non_abstained = df[~df["abstained"]].copy()
    tbl = (
        non_abstained.groupby("true_subcat")
        .agg(
            support=("true_subcat", "count"),
            accuracy=("correct", "mean"),
            avg_conf_correct=("confidence", lambda s: s[non_abstained.loc[s.index, "correct"]].mean()),
            avg_conf_wrong=("confidence", lambda s: s[~non_abstained.loc[s.index, "correct"]].mean()),
        )
        .round(3)
        .sort_values("support", ascending=False)
    )
    abstain_rate = df.groupby("true_subcat")["abstained"].mean().rename("abstain_rate").round(3)
    tbl = tbl.join(abstain_rate)
    return tbl


def _summary(df: pd.DataFrame, label: str) -> None:
    total = len(df)
    n_abstained = df["abstained"].sum()
    non_abs = df[~df["abstained"]]
    acc = non_abs["correct"].mean() if len(non_abs) > 0 else float("nan")
    coverage = (total - n_abstained) / total

    # accuracy on subcats that are in the model's label set
    in_model = df[df["in_model"] & ~df["abstained"]]
    acc_in_model = in_model["correct"].mean() if len(in_model) > 0 else float("nan")

    log.info("\n══ %s ══════════════════════════════════════", label)
    log.info("  Total test rows:    %d", total)
    log.info("  Abstained:          %d  (%.1f%%)", n_abstained, 100 * n_abstained / total)
    log.info("  Coverage:           %.1f%%  (predicted something)", 100 * coverage)
    log.info("  Accuracy (non-abs): %.4f", acc)
    log.info("  Accuracy (in-model subcats, non-abs): %.4f", acc_in_model)
    log.info("  Avg confidence (non-abs): %.4f", non_abs["confidence"].mean() if len(non_abs) > 0 else float("nan"))

    log.info("\n  Calibration (confidence → accuracy):")
    cal = _calibration_table(df)
    log.info("\n%s", cal.to_string())

    log.info("\n  Per-subcategory breakdown (top 30 by support):")
    per_sub = _per_subcat_table(df)
    log.info("\n%s", per_sub.head(30).to_string())


def run(model_variant: str = "both") -> None:
    df, flat_cats = load_and_clean("subcategory")
    _, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_CATEGORY].values,
        random_state=RANDOM_SEED,
    )
    log.info("Test rows: %d", len(df_test))

    results = {}

    if model_variant in ("floored", "both"):
        log.info("\n\n━━ Evaluating FLOORED models (support >= 50) ━━")
        df_floored = _evaluate_one_model_set(df_test, SUBCAT_MODEL_DIR, "floored")
        _summary(df_floored, "Floored models (MIN_SUBCAT_SUPPORT=50)")
        results["floored"] = df_floored

    if model_variant in ("full", "both"):
        if not (SUBCAT_FULL_MODEL_DIR / "_index.joblib").exists():
            log.warning("Full models not found at %s — run train_subcategory_full.py first", SUBCAT_FULL_MODEL_DIR)
        else:
            log.info("\n\n━━ Evaluating FULL models (no support floor) ━━")
            df_full = _evaluate_one_model_set(df_test, SUBCAT_FULL_MODEL_DIR, "full")
            _summary(df_full, "Full models (no support floor)")
            results["full"] = df_full

    if len(results) == 2:
        log.info("\n\n━━ Side-by-side comparison ━━")
        rows = []
        for variant, df_r in results.items():
            non_abs = df_r[~df_r["abstained"]]
            rows.append({
                "variant": variant,
                "total_test_rows": len(df_r),
                "abstained": df_r["abstained"].sum(),
                "abstain_rate": round(df_r["abstained"].mean(), 4),
                "accuracy_non_abstained": round(non_abs["correct"].mean(), 4) if len(non_abs) > 0 else float("nan"),
                "avg_confidence": round(non_abs["confidence"].mean(), 4) if len(non_abs) > 0 else float("nan"),
            })
        cmp = pd.DataFrame(rows).set_index("variant")
        log.info("\n%s", cmp.to_string())

    # Save detailed results to CSV for further inspection
    for variant, df_r in results.items():
        out = Path(__file__).parent / "data" / f"confidence_analysis_{variant}.csv"
        df_r.to_csv(out, index=False)
        log.info("\nDetailed row-level results saved → %s", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["floored", "full", "both"],
        default="both",
        help="Which model set to evaluate (default: both)",
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
