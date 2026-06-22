"""Stage 0 — dump label counts for human review.

Run once:
    python audit.py

Outputs (in data/audit/):
    category_counts.csv
    subcategory_counts.csv
    category_subcategory_counts.csv
    flat_categories.csv          # categories with a single subcategory
    label_decisions_template.csv # pre-filled template; human tags REAL/TRASHBIN/DEFAULT/FLAT
"""
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow running directly from the pipeline/ directory
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    AUDIT_DIR,
    DROP_FIELDS,
    LABEL_DECISIONS_PATH,
    RAW_DATA_PATH,
    TARGET_CATEGORY,
    TARGET_SERVICE,
    TARGET_SUBCAT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_raw() -> pd.DataFrame:
    path = RAW_DATA_PATH
    log.info("Loading raw data from %s", path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    log.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    return df


def run_audit(df: pd.DataFrame) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # ── per-field value counts ──────────────────────────────────────────────
    for field, fname in [
        (TARGET_SERVICE, "service_counts.csv"),
        (TARGET_CATEGORY, "category_counts.csv"),
        (TARGET_SUBCAT, "subcategory_counts.csv"),
    ]:
        vc = df[field].value_counts(dropna=False).reset_index()
        vc.columns = [field, "count"]
        vc.to_csv(AUDIT_DIR / fname, index=False)
        log.info("%s: %d unique values", field, vc[field].nunique())

    # ── (category, subcategory) pair counts ─────────────────────────────────
    pair_counts = (
        df.groupby([TARGET_CATEGORY, TARGET_SUBCAT], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    pair_counts.to_csv(AUDIT_DIR / "category_subcategory_counts.csv", index=False)
    log.info("(category, subcategory) pairs: %d", len(pair_counts))

    # ── flat categories (exactly one distinct subcategory) ──────────────────
    subcat_per_cat = (
        df.groupby(TARGET_CATEGORY)[TARGET_SUBCAT]
        .nunique()
        .reset_index(name="n_subcategories")
    )
    flat = subcat_per_cat[subcat_per_cat["n_subcategories"] == 1].copy()
    flat = flat.merge(
        df[[TARGET_CATEGORY, TARGET_SUBCAT]].drop_duplicates(),
        on=TARGET_CATEGORY,
    )
    flat.to_csv(AUDIT_DIR / "flat_categories.csv", index=False)
    log.info("Flat categories (single subcategory): %d", len(flat))
    for _, row in flat.iterrows():
        log.info("  FLAT  %s → %s", row[TARGET_CATEGORY], row[TARGET_SUBCAT])

    # ── label_decisions template ─────────────────────────────────────────────
    # Only generate if the file does not already exist (avoid overwriting human work)
    if LABEL_DECISIONS_PATH.exists():
        log.info(
            "label_decisions.csv already exists at %s — skipping template generation",
            LABEL_DECISIONS_PATH,
        )
        return

    # Build a template with every (category, subcategory) pair
    template = pair_counts.copy()
    template["tag"] = ""  # human fills with REAL | TRASHBIN | DEFAULT | FLAT

    # Pre-fill obvious cases:
    # 1. Flat categories → FLAT
    flat_cats = set(flat[TARGET_CATEGORY].tolist())
    mask_flat = template[TARGET_CATEGORY].isin(flat_cats)
    template.loc[mask_flat, "tag"] = "FLAT"

    # 2. Same-name placeholder (subcategory == category) → DEFAULT
    mask_same = template[TARGET_CATEGORY] == template[TARGET_SUBCAT]
    template.loc[mask_same & (template["tag"] == ""), "tag"] = "DEFAULT"

    # 3. Literal "Altro" → DEFAULT
    mask_altro = template[TARGET_SUBCAT].str.strip().str.lower() == "altro"
    template.loc[mask_altro & (template["tag"] == ""), "tag"] = "DEFAULT"

    LABEL_DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(LABEL_DECISIONS_PATH, index=False)
    needs_review = (template["tag"] == "").sum()
    log.info(
        "label_decisions template written to %s",
        LABEL_DECISIONS_PATH,
    )
    log.info(
        "  Pre-filled: %d rows (FLAT=%d, DEFAULT=%d). Rows needing human tag: %d",
        (template["tag"] != "").sum(),
        (template["tag"] == "FLAT").sum(),
        (template["tag"] == "DEFAULT").sum(),
        needs_review,
    )
    log.info("  Open the CSV and tag every blank row as REAL | TRASHBIN | DEFAULT")


def main() -> None:
    df = load_raw()
    run_audit(df)
    log.info("Audit complete. Outputs in %s", AUDIT_DIR)


if __name__ == "__main__":
    main()
