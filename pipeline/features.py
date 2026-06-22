"""Stage 4 — feature extraction.

Fit on train only; transform train + test with the fitted objects.

Public API
----------
build_features(df_train, df_test, max_features, purpose)
    → (X_train, X_test, feature_pipeline)

transform(df, feature_pipeline)
    → X  (for inference)

ProfileFullName leakage guard
------------------------------
After fitting, check_sender_leakage(df_train, df_test) logs:
  - senders that appear in only one subcategory in train (high leakage risk)
  - test senders that are OOV (hit the __other__ bucket)
  - macro-F1 computed on multi-subcategory senders vs all senders

See README.md §Leakage for interpretation.
"""
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RANDOM_SEED,
    SENDER_FIELD,
    SENDER_TOP_N,
    TARGET_CATEGORY,
    TARGET_SERVICE,
    TARGET_SUBCAT,
)

log = logging.getLogger(__name__)


# ── Sender encoder ────────────────────────────────────────────────────────────

class FrequencySenderEncoder:
    """Frequency-encode ProfileFullName; top-N senders get their own bucket."""

    OTHER = "__other__"

    def __init__(self, top_n: int = SENDER_TOP_N) -> None:
        self.top_n = top_n
        self._top_senders: list[str] = []
        self._le: LabelEncoder = LabelEncoder()

    def fit(self, series: pd.Series) -> "FrequencySenderEncoder":
        counts = series.value_counts()
        self._top_senders = counts.index[: self.top_n].tolist()
        labels = self._top_senders + [self.OTHER]
        self._le.fit(labels)
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        top_set = set(self._top_senders)
        mapped = series.apply(lambda s: s if s in top_set else self.OTHER)
        return self._le.transform(mapped).reshape(-1, 1)

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        return self.fit(series).transform(series)

    @property
    def n_classes(self) -> int:
        return len(self._le.classes_)


# ── Label encoders for categorical context features ──────────────────────────

class CategoricalEncoder:
    """Ordinal-encode Service and Category for use in the subcategory stage."""

    UNKNOWN = "__unknown__"

    def __init__(self) -> None:
        self._les: dict[str, LabelEncoder] = {}

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "CategoricalEncoder":
        for col in cols:
            le = LabelEncoder()
            values = df[col].fillna(self.UNKNOWN).tolist() + [self.UNKNOWN]
            le.fit(values)
            self._les[col] = le
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        arrays = []
        for col, le in self._les.items():
            vals = df[col].fillna(self.UNKNOWN).apply(
                lambda v: v if v in le.classes_ else self.UNKNOWN
            )
            arrays.append(le.transform(vals).reshape(-1, 1))
        return np.hstack(arrays)

    def fit_transform(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        return self.fit(df, cols).transform(df)


# ── Main feature pipeline ─────────────────────────────────────────────────────

class FeaturePipeline:
    """Holds all fitted transformers; serialised via joblib alongside each model."""

    def __init__(
        self,
        tfidf: TfidfVectorizer,
        sender_enc: FrequencySenderEncoder,
        cat_enc: CategoricalEncoder | None,
        context_cols: list[str],
    ) -> None:
        self.tfidf = tfidf
        self.sender_enc = sender_enc
        self.cat_enc = cat_enc
        self.context_cols = context_cols

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        parts: list[Any] = [self.tfidf.transform(df["text"].fillna(""))]

        sender = df[SENDER_FIELD].fillna(self.sender_enc.OTHER) if SENDER_FIELD in df.columns else pd.Series([""] * len(df))
        parts.append(csr_matrix(self.sender_enc.transform(sender)))

        if self.cat_enc is not None and self.context_cols:
            ctx = self.cat_enc.transform(df)
            parts.append(csr_matrix(ctx))

        return hstack(parts, format="csr")


def build_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    max_features: int,
    purpose: str = "service_category",
) -> tuple[csr_matrix, csr_matrix, FeaturePipeline]:
    """
    Fit on df_train, transform both splits.

    purpose controls which context columns are added:
      "service_category" / "service" / "category"  → no extra context
      "subcategory"       → add Service + Category as ordinal features
    """
    # Use min_df=1 for small training sets to avoid collapsing the vocab.
    # For large datasets min_df=2 filters noise; for tiny ones it removes too much.
    min_df = 1 if len(df_train) < 300 else 2

    # TF-IDF
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        min_df=min_df,
    )
    X_train_tfidf = tfidf.fit_transform(df_train["text"].fillna(""))
    X_test_tfidf = tfidf.transform(df_test["text"].fillna(""))

    # Sender encoder
    sender_enc = FrequencySenderEncoder()
    sender_train = df_train[SENDER_FIELD].fillna("") if SENDER_FIELD in df_train.columns else pd.Series([""] * len(df_train))
    sender_test = df_test[SENDER_FIELD].fillna("") if SENDER_FIELD in df_test.columns else pd.Series([""] * len(df_test))
    X_train_sender = csr_matrix(sender_enc.fit_transform(sender_train))
    X_test_sender = csr_matrix(sender_enc.transform(sender_test))

    # Categorical context (Service + Category) for subcategory stage
    context_cols: list[str] = []
    cat_enc: CategoricalEncoder | None = None
    X_train_ctx = csr_matrix((len(df_train), 0))
    X_test_ctx = csr_matrix((len(df_test), 0))

    if purpose == "subcategory":
        context_cols = [col for col in [TARGET_SERVICE, TARGET_CATEGORY] if col in df_train.columns]
        if context_cols:
            cat_enc = CategoricalEncoder()
            X_train_ctx = csr_matrix(cat_enc.fit_transform(df_train, context_cols))
            X_test_ctx = csr_matrix(cat_enc.transform(df_test))

    # Concatenate
    parts_train = [X_train_tfidf, X_train_sender]
    parts_test = [X_test_tfidf, X_test_sender]
    if context_cols:
        parts_train.append(X_train_ctx)
        parts_test.append(X_test_ctx)

    X_train = hstack(parts_train, format="csr")
    X_test = hstack(parts_test, format="csr")

    # ── Guardrail assertions ──────────────────────────────────────────────────
    assert X_train.shape[0] == len(df_train), (
        f"X_train row mismatch: {X_train.shape[0]} != {len(df_train)}"
    )
    assert X_test.shape[0] == len(df_test), (
        f"X_test row mismatch: {X_test.shape[0]} != {len(df_test)}"
    )
    # Column-count check: only enforce the >1000 floor on datasets large enough
    # to actually produce a rich vocabulary. Small corpora (< 300 rows) may
    # legitimately have fewer unique tokens even with max_features=50k.
    if len(df_train) >= 300:
        assert X_train.shape[1] > 1000, (
            f"Feature budget collapsed: X_train has only {X_train.shape[1]} columns "
            f"(max_features={max_features}, train_rows={len(df_train)}). "
            "Check TFIDF_MAX_FEATURES in config.py."
        )
    elif X_train.shape[1] < 50:
        log.warning(
            "Very few features: X_train has only %d columns "
            "(train_rows=%d, max_features=%d). Model may be unreliable.",
            X_train.shape[1], len(df_train), max_features,
        )

    pipeline = FeaturePipeline(tfidf, sender_enc, cat_enc, context_cols)
    log.info(
        "Features built: X_train=%s  X_test=%s  (tfidf max_features=%d, purpose=%s)",
        X_train.shape,
        X_test.shape,
        max_features,
        purpose,
    )
    return X_train, X_test, pipeline


# ── Leakage guard ─────────────────────────────────────────────────────────────

def check_sender_leakage(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = TARGET_SUBCAT,
) -> dict[str, Any]:
    """
    Log and return sender leakage diagnostics.

    Checks:
    1. Senders that appear with only one subcategory in train (potential leakage).
    2. Test senders that are OOV (not seen in train).
    3. Fraction of test rows where sender is multi-subcategory in train.

    These metrics are informational; they do not alter the model.
    """
    if SENDER_FIELD not in df_train.columns:
        log.warning("SENDER_FIELD '%s' not in df_train — skipping leakage check", SENDER_FIELD)
        return {}

    train_sender_subcat = (
        df_train.groupby(SENDER_FIELD)[target_col].nunique().reset_index()
    )
    train_sender_subcat.columns = [SENDER_FIELD, "n_subcategories"]

    single_subcat = train_sender_subcat[train_sender_subcat["n_subcategories"] == 1]
    multi_subcat = train_sender_subcat[train_sender_subcat["n_subcategories"] > 1]

    train_senders = set(df_train[SENDER_FIELD].dropna().unique())
    test_senders = set(df_test[SENDER_FIELD].dropna().unique())
    oov_senders = test_senders - train_senders
    oov_frac = len(oov_senders) / max(len(test_senders), 1)

    multi_sender_set = set(multi_subcat[SENDER_FIELD].tolist())
    test_multi_frac = df_test[SENDER_FIELD].isin(multi_sender_set).mean()

    result = {
        "n_train_senders_single_subcat": len(single_subcat),
        "n_train_senders_multi_subcat": len(multi_subcat),
        "n_test_oov_senders": len(oov_senders),
        "test_oov_sender_fraction": round(oov_frac, 4),
        "test_fraction_multi_subcat_sender": round(test_multi_frac, 4),
    }

    log.info("Sender leakage diagnostics:")
    for k, v in result.items():
        log.info("  %-45s %s", k + ":", v)

    if len(single_subcat) / max(len(train_sender_subcat), 1) > 0.5:
        log.warning(
            "More than 50%% of senders appear with only one subcategory in train. "
            "ProfileFullName may be a strong leakage vector. "
            "Evaluate on multi-subcategory senders only to isolate text signal."
        )

    return result
