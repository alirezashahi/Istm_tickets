# ITSM Ticket Routing Pipeline

Automatic three-stage classification of SCM support tickets:
**Service → Category → Subcategory**

Trained on Italian/English helpdesk tickets exported from Ivanti. Each stage is a
LinearSVC with three dedicated TF-IDF vectorizers and calibrated confidence scores.

---

## Results

| Stage | Accuracy | Macro F1 |
|---|---:|---:|
| Service | **96.68%** | 0.92 |
| Category | **87.00%** | 0.63 |
| Subcategory — true-routed | **95.05%** | 0.94 |
| Subcategory — end-to-end | **84.61%** | 0.74 |

Full analysis: [`pipeline/service_analysis.md`](pipeline/service_analysis.md) · [`pipeline/category_analysis.md`](pipeline/category_analysis.md) · [`pipeline/subcategory_analysis.md`](pipeline/subcategory_analysis.md)

---

## How it works

Each stage shares the same architecture:

```
ProfileFullName  →  TF-IDF (5k features)  ─┐
Subject          →  TF-IDF (15k features) ──┼─► LinearSVC ─► CalibratedClassifierCV
Symptom          →  TF-IDF (35k features) ─┘
```

**Stage 1 — Service**: global model, predicts `Application` or `Infrastructure`.

**Stage 2 — Category**: global model, predicts one of 41 trained categories independently
of the service prediction.

**Stage 3 — Subcategory**: one model per category (18 trained). If the predicted category
has no trained model, returns `"unspecified (review)"`.

**Flat categories** (14): no model needed — subcategory equals the category name by rule.

A noise filter removes ~33% of training rows whose `Category` or `Subcategory` label
matches `(?i)(cms|altro|other|z-other)` — catch-all buckets that would pollute the
decision boundary.

---

## Directory structure

```
pipeline/
  config.py                 constants, paths, thresholds
  clean.py                  data loading, label filtering, text cleaning
  train_service.py          Stage 1: train Service model
  train_category.py         Stage 2: train Category model
  train_subcategory.py      Stage 3: train per-category Subcategory models
  evaluate.py               per-stage + end-to-end evaluation
  pipeline.py               inference: loads all models, runs full prediction
  api.py                    FastAPI HTTP service (wraps pipeline.py)
  audit.py                  dump label counts for human review
  features.py               legacy feature pipeline (reference only)
  data/
    label_decisions.csv     human-audited label tags (TRASHBIN / FLAT / DEFAULT / REAL)
  models/                   ← not in git, rebuilt by training scripts
    service_model.joblib
    service_model_calibrated.joblib
    service_transformers.joblib
    category_model.joblib
    category_model_calibrated.joblib
    category_transformers.joblib
    subcategory/<category>.joblib
  service_analysis.md       service model performance analysis
  category_analysis.md      category model performance analysis
  subcategory_analysis.md   subcategory model performance analysis
```

---

## Setup

```bash
pip install scikit-learn pandas numpy scipy joblib matplotlib seaborn fastapi uvicorn
```

Place the raw ticket CSV export in the project root. The path is configured in
`pipeline/config.py` (`RAW_DATA_PATH`).

---

## Rebuild models

Run in order (each script is self-contained):

```bash
cd pipeline
python train_service.py
python train_category.py
python train_subcategory.py
```

Models are saved to `pipeline/models/`. The CSV is never committed — it contains PII.
`label_decisions.csv` is the only data file in the repo; it controls which labels are
treated as noise, flat, or real.

---

## Evaluate

```bash
cd pipeline
python evaluate.py
```

Reports service, category, and subcategory metrics in both true-routed and end-to-end
(predicted routing) modes.

---

## Inference

```python
import sys
sys.path.insert(0, "pipeline")
from pipeline import Pipeline

p = Pipeline.load()
result = p.predict(
    subject="Workflow fermi nella worklist",
    symptom="Due workflow sembrano bloccati nella coda approvazioni",
    sender="CARLO VANNUCCI",
)

print(result.service)        # "Application"
print(result.category)       # "34-PLM"
print(result.subcategory)    # "Workflow"
print(result.confidences)    # {"service": 0.99, "category": 0.91, "subcategory": 0.84}
print(result.is_flat)        # False
```

CLI:

```bash
cd pipeline
python pipeline.py --subject "Workflow bloccato" --symptom "Worklist non risponde" --sender "MARIO ROSSI"
```

---

## REST API

Start the server:

```bash
cd pipeline
uvicorn api:app --host 0.0.0.0 --port 8000
```

The pipeline loads once at startup. Interactive docs are available at
`http://localhost:8000/docs`.

**Predict endpoint** — `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "incident_number": "INC0012345",
    "subject": "Workflow bloccato",
    "symptom": "Worklist non risponde",
    "sender": "MARIO ROSSI"
  }'
```

Response:

```json
{
  "incident_number": "INC0012345",
  "service": "Application",
  "category": "34-PLM",
  "subcategory": "Workflow",
  "confidences": {
    "service": 0.984,
    "category": 0.091,
    "subcategory": 0.870
  },
  "is_flat": false
}
```

**Request fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `incident_number` | string | No | Echoed back in the response for correlation |
| `subject` | string | No | Ticket subject line |
| `symptom` | string | No | Ticket description / symptom body |
| `sender` | string | No | `ProfileFullName` of the requester |
| `category_hint` | string | No | Skip category model and route directly to this category |

**Health check** — `GET /health`

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## Key configuration (`pipeline/config.py`)

| Constant | Default | Effect |
|---|---|---|
| `MIN_SUBCAT_SUPPORT` | 50 | Subcategories with fewer rows are excluded from training |
| `TEST_SIZE` | 0.20 | Stratified hold-out fraction used by all training scripts |
| `RANDOM_SEED` | 42 | All randomness seeded for reproducibility |
