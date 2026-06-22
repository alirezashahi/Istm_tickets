"""Inference glue — predict Service → Category → Subcategory for new tickets.

Usage
-----
from pipeline import Pipeline

p = Pipeline.load()
result = p.predict(subject="...", symptom="...", sender="...", category_hint=None)
# result = {"service": ..., "category": ..., "subcategory": ..., "confidences": {...}}

Or from the CLI:
    python pipeline.py --subject "Workflow bloccato" --symptom "..."
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack as sp_hstack

sys.path.insert(0, str(Path(__file__).parent))

from clean import clean_text
from config import (
    ABSTAIN_LABEL,
    CATEGORY_MODEL_CALIBRATED_PATH,
    CATEGORY_MODEL_PATH,
    CATEGORY_TRANSFORMERS_PATH,
    SENDER_FIELD,
    SERVICE_MODEL_CALIBRATED_PATH,
    SERVICE_MODEL_PATH,
    SERVICE_TRANSFORMERS_PATH,
    SUBCAT_MODEL_DIR,
    TARGET_CATEGORY,
    TARGET_SERVICE,
    TARGET_SUBCAT,
)

log = logging.getLogger(__name__)


def _safe_name(category: str) -> str:
    return re.sub(r"[^\w\-]", "_", category)


@dataclass
class PredictionResult:
    service: str
    category: str
    subcategory: str
    confidences: dict[str, float] = field(default_factory=dict)
    abstained: bool = False
    is_flat: bool = False


class Pipeline:
    """Loaded inference pipeline — holds all model artefacts in memory."""

    def __init__(
        self,
        svc_model: Any,
        svc_calibrated: Any,
        svc_transformers: dict,
        cat_model: Any,
        cat_calibrated: Any,
        cat_transformers: dict,
        subcat_index: dict,
        subcat_artefacts: dict[str, dict],
    ) -> None:
        self._svc_model = svc_model
        self._svc_calibrated = svc_calibrated
        self._svc_transformers = svc_transformers
        self._cat_model = cat_model
        self._cat_calibrated = cat_calibrated
        self._cat_transformers = cat_transformers
        self._subcat_index = subcat_index
        self._subcat = subcat_artefacts
        self._flat_cats: set[str] = set(subcat_index.get("flat_categories", []))

    @classmethod
    def load(cls) -> "Pipeline":
        """Load all models from disk."""
        if not SERVICE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Service model not found: {SERVICE_MODEL_PATH}\n"
                "Run train_service.py first."
            )
        for path, label in [
            (SERVICE_MODEL_PATH, "Service model"),
            (SERVICE_MODEL_CALIBRATED_PATH, "Calibrated service model"),
            (SERVICE_TRANSFORMERS_PATH, "Service transformers"),
            (CATEGORY_MODEL_PATH, "Category model"),
            (CATEGORY_MODEL_CALIBRATED_PATH, "Calibrated category model"),
            (CATEGORY_TRANSFORMERS_PATH, "Category transformers"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} not found: {path}\n"
                    "Run train_service.py / train_category.py first."
                )

        svc_model = joblib.load(SERVICE_MODEL_PATH)
        svc_calibrated = joblib.load(SERVICE_MODEL_CALIBRATED_PATH)
        svc_transformers = joblib.load(SERVICE_TRANSFORMERS_PATH)
        cat_model = joblib.load(CATEGORY_MODEL_PATH)
        cat_calibrated = joblib.load(CATEGORY_MODEL_CALIBRATED_PATH)
        cat_transformers = joblib.load(CATEGORY_TRANSFORMERS_PATH)

        index_path = SUBCAT_MODEL_DIR / "_index.joblib"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Subcategory index not found: {index_path}\n"
                "Run train_subcategory.py first."
            )
        index = joblib.load(index_path)

        subcat_arts: dict[str, dict] = {}
        for cat in index.get("trained_categories", []):
            path = SUBCAT_MODEL_DIR / f"{_safe_name(cat)}.joblib"
            if path.exists():
                subcat_arts[cat] = joblib.load(path)
            else:
                log.warning("Subcategory model missing for category: %s", cat)

        return cls(svc_model, svc_calibrated, svc_transformers,
                   cat_model, cat_calibrated, cat_transformers,
                   index, subcat_arts)

    def predict(
        self,
        subject: str = "",
        symptom: str = "",
        sender: str = "",
        category_hint: str | None = None,
    ) -> PredictionResult:
        """
        Predict service, category, and subcategory for a single ticket.

        category_hint: if provided (e.g. from the ticket form), skips the
                       category model and goes straight to subcategory.
        """
        # ── 1. Build a one-row DataFrame ──────────────────────────────────────
        row = pd.DataFrame(
            [{
                "text": (clean_text(subject) + " " + clean_text(symptom)).strip(),
                "Subject": subject,
                "Symptom": symptom,
                SENDER_FIELD: sender or "",
                TARGET_SERVICE: "",
                TARGET_CATEGORY: category_hint or "",
            }]
        )

        confidences: dict[str, float] = {}

        # ── 2. Service ────────────────────────────────────────────────────────

        X_svc_name = self._svc_transformers["tfidf_name"].transform([sender or ""])
        X_svc_subj = self._svc_transformers["tfidf_subject"].transform([clean_text(subject)])
        X_svc_symp = self._svc_transformers["tfidf_symptom"].transform([clean_text(symptom)])
        X_svc = sp_hstack([X_svc_name, X_svc_subj, X_svc_symp])
        service = self._svc_model.predict(X_svc)[0]
        if service in self._svc_calibrated.classes_:
            svc_proba = self._svc_calibrated.predict_proba(X_svc)[0]
            svc_idx = list(self._svc_calibrated.classes_).index(service)
            confidences["service"] = round(float(svc_proba[svc_idx]), 4)
        else:
            confidences["service"] = 0.0

        # Update row with predicted service
        row[TARGET_SERVICE] = service

        # ── 3. Category ───────────────────────────────────────────────────────
        if category_hint:
            category = category_hint
            confidences["category"] = 1.0
        else:
    
            X_name = self._cat_transformers["tfidf_name"].transform([row[SENDER_FIELD].iloc[0]])
            X_subj = self._cat_transformers["tfidf_subject"].transform([clean_text(subject)])
            X_symp = self._cat_transformers["tfidf_symptom"].transform([clean_text(symptom)])
            X_cat = sp_hstack([X_name, X_subj, X_symp])
            category = self._cat_model.predict(X_cat)[0]
            # Use calibrated model for confidence
            if category in self._cat_calibrated.classes_:
                cat_proba = self._cat_calibrated.predict_proba(X_cat)[0]
                cat_idx = list(self._cat_calibrated.classes_).index(category)
                confidences["category"] = round(float(cat_proba[cat_idx]), 4)
            else:
                confidences["category"] = 0.0

        # Update row with predicted category
        row[TARGET_CATEGORY] = category

        # ── 4. Subcategory ────────────────────────────────────────────────────
        is_flat = category in self._flat_cats
        abstained = False

        if is_flat:
            # Rule: flat category → subcategory = category name
            subcategory = category

        elif category in self._subcat:
            sub_model_data = self._subcat[category]
            X_sub_name = sub_model_data["transformers"]["tfidf_name"].transform([sender or ""])
            X_sub_subj = sub_model_data["transformers"]["tfidf_subject"].transform([clean_text(subject)])
            X_sub_symp = sub_model_data["transformers"]["tfidf_symptom"].transform([clean_text(symptom)])
            X_sub = sp_hstack([X_sub_name, X_sub_subj, X_sub_symp])
            subcategory = sub_model_data["model"].predict(X_sub)[0]
            sub_proba = sub_model_data["calibrated"].predict_proba(X_sub)[0]
            sub_idx = list(sub_model_data["calibrated"].classes_).index(subcategory)
            top_conf = float(sub_proba[sub_idx])
            confidences["subcategory"] = round(top_conf, 4)

        else:
            subcategory = ABSTAIN_LABEL
            abstained = True
            confidences["subcategory"] = 0.0

        return PredictionResult(
            service=service,
            category=category,
            subcategory=subcategory,
            confidences=confidences,
            abstained=abstained,
            is_flat=is_flat,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Predict service/category/subcategory for a ticket")
    parser.add_argument("--subject", default="", help="Ticket subject")
    parser.add_argument("--symptom", default="", help="Ticket symptom / description")
    parser.add_argument("--sender", default="", help="Sender name (ProfileFullName)")
    parser.add_argument("--category", default=None, help="Category hint (optional)")
    args = parser.parse_args()

    p = Pipeline.load()
    result = p.predict(
        subject=args.subject,
        symptom=args.symptom,
        sender=args.sender,
        category_hint=args.category,
    )

    print(f"\nService     : {result.service}  (conf={result.confidences.get('service', '?'):.3f})")
    print(f"Category    : {result.category}  (conf={result.confidences.get('category', '?'):.3f})")
    subcat_conf = result.confidences.get("subcategory", "N/A")
    conf_str = f"{subcat_conf:.3f}" if isinstance(subcat_conf, float) else subcat_conf
    flag = " [FLAT]" if result.is_flat else (" [NO MODEL]" if result.abstained else "")
    print(f"Subcategory : {result.subcategory}  (conf={conf_str}){flag}")


if __name__ == "__main__":
    main()
