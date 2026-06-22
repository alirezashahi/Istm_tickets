"""Stages 1–3 — load, integrity checks, label filtering, text cleaning.

Public API
----------
load_and_clean(purpose)  → pd.DataFrame
    purpose = "service_category"  : rows with REAL/FLAT categories (includes DEFAULT subcat rows)
    purpose = "subcategory"        : only REAL subcategories, flat cats excluded
    purpose = "all"                : everything after structural drops / integrity

clean_text(s)  → str
    Idempotent, used identically at train time and inference time.
"""
import logging
import re
import sys
from pathlib import Path
from typing import Literal

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DROP_FIELDS,
    LABEL_DECISIONS_PATH,
    RAW_DATA_PATH,
    TAG_DEFAULT,
    TAG_FLAT,
    TAG_REAL,
    TAG_TRASHBIN,
    TARGET_CATEGORY,
    TARGET_SERVICE,
    TARGET_SUBCAT,
    TEXT_FIELDS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Text cleaning patterns ────────────────────────────────────────────────────

# SCM email footer line
_RE_SCM_FOOTER = re.compile(r"www\.scmgroup\.com[^\n]*", re.IGNORECASE)

# Phone lines (e.g. "Phone: +39 0541 700522")
_RE_PHONE_LINE = re.compile(
    r"(?:phone|tel|fax|mobile|cell)\s*:?\s*[\+\d\s\(\)\-\.]{6,}", re.IGNORECASE
)

# Address blocks: lines that look like street/city/country/postcode
_RE_ADDRESS_LINE = re.compile(
    r"\b(?:via|viale|piazza|corso|str\.|street|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?)"
    r"[^\n]{0,60}",
    re.IGNORECASE,
)

# "This email is from an unusual…" disclaimer lines
_RE_DISCLAIMER = re.compile(
    r"(?:this (?:e-?mail|message) (?:is|was|comes from|originated from)[^\n]*"
    r"|attenzione[^\n]*mittente[^\n]*"
    r"|disclaimer[^\n]*)",
    re.IGNORECASE,
)

# Inline image references [cid:…]
_RE_CID = re.compile(r"\[cid:[^\]]*\]", re.IGNORECASE)

# vCalendar blocks
_RE_VCAL = re.compile(r"BEGIN:VCALENDAR.*?END:VCALENDAR", re.DOTALL | re.IGNORECASE)

# Email addresses
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")

# URLs
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Standalone phone numbers (not yet caught by the line-level pattern)
_RE_PHONE_NUM = re.compile(r"\b[\+\d][\d\s\(\)\-\.]{7,}\b")

# Control characters (except newline/tab)
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(s: object) -> str:
    """Reusable text cleaner — call identically at train time and inference time."""
    if not isinstance(s, str) or not s.strip():
        return ""
    text = s

    # Remove vCalendar blocks first (can be multi-line)
    text = _RE_VCAL.sub(" ", text)

    # Remove [cid:…] inline image tags
    text = _RE_CID.sub(" ", text)

    # Remove disclaimers
    text = _RE_DISCLAIMER.sub(" ", text)

    # Remove SCM footer
    text = _RE_SCM_FOOTER.sub(" ", text)

    # Remove phone lines
    text = _RE_PHONE_LINE.sub(" ", text)

    # Remove address lines
    text = _RE_ADDRESS_LINE.sub(" ", text)

    # Substitute emails and URLs with tokens (preserve some signal)
    text = _RE_EMAIL.sub(" __email__ ", text)
    text = _RE_URL.sub(" __url__ ", text)

    # Remove residual phone numbers
    text = _RE_PHONE_NUM.sub(" ", text)

    # Remove control characters
    text = _RE_CONTROL.sub(" ", text)

    # Lowercase
    text = text.lower()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_raw() -> pd.DataFrame:
    path = RAW_DATA_PATH
    log.info("Loading %s", path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    log.info("  rows: %d", len(df))
    return df


def _load_label_decisions() -> pd.DataFrame:
    if not LABEL_DECISIONS_PATH.exists():
        raise FileNotFoundError(
            f"label_decisions.csv not found at {LABEL_DECISIONS_PATH}.\n"
            "Run audit.py first, then fill in the tag column."
        )
    ld = pd.read_csv(LABEL_DECISIONS_PATH)
    required = {TARGET_CATEGORY, TARGET_SUBCAT, "tag"}
    missing = required - set(ld.columns)
    if missing:
        raise ValueError(f"label_decisions.csv is missing columns: {missing}")
    # Normalise tags
    ld["tag"] = ld["tag"].str.strip().str.upper()
    return ld


def _ledger(df: pd.DataFrame, step: str) -> None:
    log.info("  [ledger] after %-40s rows=%d", step + ":", len(df))


# ── Stage 1 — load & integrity ────────────────────────────────────────────────

def _stage1_load(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unwanted columns (ignore missing ones)
    drop = [c for c in DROP_FIELDS if c in df.columns]
    df = df.drop(columns=drop)
    _ledger(df, "drop DROP_FIELDS")

    # Drop rows missing core labels
    before = len(df)
    df = df.dropna(subset=[TARGET_SERVICE, TARGET_CATEGORY, TARGET_SUBCAT])
    _ledger(df, f"dropna core labels (removed {before - len(df)})")

    # Strip whitespace from label columns
    for col in [TARGET_SERVICE, TARGET_CATEGORY, TARGET_SUBCAT]:
        df[col] = df[col].str.strip()

    return df


# ── Stage 2 — apply label decisions ──────────────────────────────────────────

def _stage2_labels(
    df: pd.DataFrame,
    ld: pd.DataFrame,
    purpose: str,
) -> tuple[pd.DataFrame, set]:
    """
    Returns (filtered_df, flat_categories).

    purpose="service"
        No label filtering — only structural drops (Stage 1) apply.
        Returns the full dataset so the service model trains on all ticket types.

    purpose="category"
        Drops TRASHBIN categories only.
        DEFAULT/FLAT subcategory rows are kept because the category model needs
        to learn to predict every non-TRASHBIN category.

    purpose="subcategory"
        Drops TRASHBIN categories + DEFAULT subcategory rows + excludes FLAT
        categories (those are handled by rule at inference).
    """
    # Build lookup: (category, subcategory) → tag
    tag_map: dict[tuple, str] = {
        (row[TARGET_CATEGORY], row[TARGET_SUBCAT]): row["tag"]
        for _, row in ld.iterrows()
    }

    # Category-level tag derived from any TRASHBIN or FLAT row for that category
    cat_tag: dict[str, str] = {}
    for (cat, _sub), tag in tag_map.items():
        if tag in (TAG_TRASHBIN, TAG_FLAT):
            cat_tag[cat] = tag

    # Identify FLAT categories (needed for subcategory purpose and inference rules)
    flat_cats: set[str] = {cat for cat, tag in cat_tag.items() if tag == TAG_FLAT}

    if purpose == "service":
        # No label-based filtering; return everything after structural drops
        log.info("  purpose=service: skipping label filtering")
        return df, flat_cats

    # purpose == "category" or "subcategory": drop TRASHBIN categories
    trashbin_cats = {cat for cat, tag in cat_tag.items() if tag == TAG_TRASHBIN}
    before = len(df)
    df = df[~df[TARGET_CATEGORY].isin(trashbin_cats)].copy()
    _ledger(df, f"drop TRASHBIN categories ({before - len(df)} rows removed)")

    log.info("  FLAT categories (%d): %s", len(flat_cats), sorted(flat_cats))

    if purpose == "subcategory":
        # Drop DEFAULT subcategories (Altro, Z-Others, same-name placeholders)
        default_pairs = {pair for pair, tag in tag_map.items() if tag == TAG_DEFAULT}
        before = len(df)
        df = df[
            ~df.apply(
                lambda r: (r[TARGET_CATEGORY], r[TARGET_SUBCAT]) in default_pairs,
                axis=1,
            )
        ].copy()
        _ledger(df, f"drop DEFAULT subcategories ({before - len(df)} rows removed)")

        # Exclude flat categories from subcategory training (handled by rule at inference)
        before = len(df)
        df = df[~df[TARGET_CATEGORY].isin(flat_cats)].copy()
        _ledger(df, f"exclude FLAT categories ({before - len(df)} rows removed)")

    return df, flat_cats


# ── Stage 3 — text cleaning ───────────────────────────────────────────────────

def _stage3_text(df: pd.DataFrame) -> pd.DataFrame:
    for field in TEXT_FIELDS:
        if field not in df.columns:
            df[field] = ""
        else:
            df[field] = df[field].fillna("").astype(str)

    df["text"] = df[TEXT_FIELDS[0]].map(clean_text) + " " + df[TEXT_FIELDS[1]].map(clean_text)
    df["text"] = df["text"].str.strip()
    _ledger(df, "text cleaning")
    return df


# ── Public entry point ────────────────────────────────────────────────────────

Purpose = Literal["service", "category", "subcategory", "all"]


def load_and_clean(
    purpose: Purpose = "subcategory",
) -> tuple[pd.DataFrame, set]:
    """
    Returns (df, flat_categories).

    purpose="service"
        All tickets after null drops only. No label filtering.
        Use for training the Service classifier.

    purpose="category"
        Drops TRASHBIN categories; keeps DEFAULT and FLAT rows.
        Use for training the Category classifiers.

    purpose="subcategory"
        Full filtering: TRASHBIN dropped, DEFAULT subcategory rows dropped,
        FLAT categories excluded (handled by rule at inference).
        Use for training the Subcategory classifiers.

    purpose="all"
        Structural drops only (no label filtering). Useful for inspection.
    """
    df = _load_raw()
    df = _stage1_load(df)

    if purpose == "all":
        df = _stage3_text(df)
        return df, set()

    ld = _load_label_decisions()
    df, flat_cats = _stage2_labels(df, ld, purpose)
    df = _stage3_text(df)

    log.info("Final shape for purpose='%s': %d rows", purpose, len(df))
    return df, flat_cats


if __name__ == "__main__":
    # Quick smoke test
    for p in ("service", "category", "subcategory"):
        df, flat = load_and_clean(p)
        print(f"purpose={p}: {len(df)} rows, {df[TARGET_CATEGORY].nunique()} categories")
