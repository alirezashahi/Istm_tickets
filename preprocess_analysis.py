#!/usr/bin/env python3
"""
ITSM Incident Data — Preprocessing & Exploratory Analysis
Target: Subcategory prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='viridis')
plt.rcParams['figure.figsize'] = (14, 6)

# ── 1. Load ──────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "2026-06-05_01.13.20_ItsmIncidentExport.csv"
OUT_DIR = Path(__file__).parent / "analysis_output"
OUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("LOADING DATA")
print("=" * 60)
df = pd.read_csv(DATA_PATH, dtype_backend='numpy_nullable', low_memory=False)
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{list(df.columns)}")

# ── 2. Initial glance ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA TYPES & BASIC INFO")
print("=" * 60)
print(df.dtypes.to_string())
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# ── 3. Missing values ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MISSING VALUES (per column)")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Count': missing, 'Percent': missing_pct})
missing_df = missing_df.sort_values('Percent', ascending=False)
print(missing_df.to_string())
missing_df.to_csv(OUT_DIR / "missing_values.csv")

# Visual
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
bars = ax.barh(missing_df.index, missing_df['Percent'], color='coral')
ax.set_xlabel('Missing %')
ax.set_title('% Missing Values per Column')
for b in bars:
    w = b.get_width()
    if w > 0:
        ax.text(w + 0.5, b.get_y() + b.get_height() / 2, f'{w:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "missing_values.png", dpi=150)
plt.close()
print("[SAVED] missing_values.png")

# ── 4. Target variable: Subcategory ──────────────────────────────────────────
print("\n" + "=" * 60)
print("TARGET: SUBCATEGORY")
print("=" * 60)
target_null = df['Subcategory'].isnull().sum()
print(f"Null Subcategory: {target_null} ({target_null / len(df) * 100:.2f}%)")
vc = df['Subcategory'].value_counts()
print(f"Unique Subcategories: {len(vc)}")
print("\nTop 20 Subcategories:")
print(vc.head(20).to_string())

# Plot top 30
top_n = 30
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
top_vals = vc.head(top_n)
bars = ax.barh(range(len(top_vals)), top_vals.values, color='steelblue')
ax.set_yticks(range(len(top_vals)))
ax.set_yticklabels(top_vals.index)
ax.invert_yaxis()
ax.set_xlabel('Count')
ax.set_title(f'Top {top_n} Subcategories')
for b in bars:
    ax.text(b.get_width() + 30, b.get_y() + b.get_height() / 2,
            str(int(b.get_width())), va='center', fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "subcategory_top30.png", dpi=150)
plt.close()
print("[SAVED] subcategory_top30.png")

# ── 5. Target imbalance ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TARGET IMBALANCE METRICS")
print("=" * 60)
vc_sorted = vc.sort_values(ascending=False)
cumsum = vc_sorted.cumsum()
cumsum_pct = cumsum / cumsum.iloc[-1] * 100
print(f"Top 1 class covers: {vc_sorted.iloc[0] / len(df) * 100:.2f}% of data")
print(f"Top 5 classes cover: {cumsum_pct.iloc[4]:.2f}% of data")
print(f"Top 10 classes cover: {cumsum_pct.iloc[9]:.2f}% of data")
print(f"Top 20 classes cover: {cumsum_pct.iloc[19]:.2f}% of data")
print(f"Classes with < 10 samples: {(vc < 10).sum()}")
print(f"Classes with < 50 samples: {(vc < 50).sum()}")
print(f"Classes with < 100 samples: {(vc < 100).sum()}")

# ── 6. Categorical feature analysis ─────────────────────────────────────────
cat_cols = ['Category', 'Service', 'SCMTypeOfRequest', 'CustomerLocation']
print("\n" + "=" * 60)
print("CATEGORICAL FEATURES ANALYSIS")
print("=" * 60)
for col in cat_cols:
    if col not in df.columns:
        continue
    nulls = df[col].isnull().sum()
    uniq = df[col].nunique()
    top_val = df[col].value_counts().iloc[0] if uniq > 0 else 0
    top_name = df[col].value_counts().index[0] if uniq > 0 else ''
    print(f"\n--- {col} ---")
    print(f"  Nulls: {nulls} ({nulls / len(df) * 100:.2f}%)")
    print(f"  Unique: {uniq}")
    print(f"  Top: '{top_name}' -> {top_val} ({top_val / len(df) * 100:.2f}%)")

# ── 7. Text feature analysis ─────────────────────────────────────────────────
text_cols = ['Subject', 'Symptom', 'SCMTitle', 'ProfileFullName', 'Email', 'IncidentNumber']
print("\n" + "=" * 60)
print("TEXT FEATURES ANALYSIS")
print("=" * 60)
for col in text_cols:
    if col not in df.columns:
        continue
    nulls = df[col].isnull().sum()
    filled = df[col].dropna()
    lengths = filled.astype(str).str.len()
    words = filled.astype(str).str.split().str.len()
    print(f"\n--- {col} ---")
    print(f"  Nulls: {nulls} ({nulls / len(df) * 100:.2f}%)")
    print(f"  Char length: mean={lengths.mean():.1f}  std={lengths.std():.1f}  "
          f"min={lengths.min()}  max={lengths.max()}")
    print(f"  Word count:  mean={words.mean():.1f}  std={words.std():.1f}  "
          f"min={words.min()}  max={words.max()}")
    print(f"  Unique values: {filled.nunique()}")

# ── 8. Correlation between categorical features (Cramér's V) ─────────────────
print("\n" + "=" * 60)
print("CRAMÉR'S V CORRELATION (categorical-categorical)")
print("=" * 60)

def cramers_v(series_a, series_b):
    """Compute Cramér's V statistic for categorical-categorical association."""
    from scipy.stats import chi2_contingency
    confreq = pd.crosstab(series_a, series_b)
    if confreq.size == 0:
        return np.nan
    chi2, _, _, _ = chi2_contingency(confreq)
    n = confreq.sum().sum()
    phi2 = chi2 / n
    r, k = confreq.shape
    v = np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else np.nan
    return v

# Pairs of interest (including target)
corr_cols = [
    c for c in ['Category', 'Service', 'SCMTypeOfRequest', 'CustomerLocation', 'Subcategory']
    if c in df.columns
]
corr_matrix = pd.DataFrame(np.nan, index=corr_cols, columns=corr_cols)
for c1 in corr_cols:
    for c2 in corr_cols:
        if c1 == c2:
            corr_matrix.loc[c1, c2] = 1.0
        else:
            corr_matrix.loc[c1, c2] = cramers_v(df[c1].astype(str), df[c2].astype(str))

print(corr_matrix.round(3).to_string())
corr_matrix.round(3).to_csv(OUT_DIR / "cramers_v_correlation.csv")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix.astype(float), annot=True, fmt='.2f', cmap='YlOrRd',
            vmin=0, vmax=1, ax=ax, linewidths=0.5)
ax.set_title("Cramér's V — Categorical Feature Association")
plt.tight_layout()
plt.savefig(OUT_DIR / "cramers_v_heatmap.png", dpi=150)
plt.close()
print("[SAVED] cramers_v_heatmap.png")

# ── 9. Category → Subcategory relationship ───────────────────────────────────
print("\n" + "=" * 60)
print("CATEGORY → SUBCATEGORY MAPPING")
print("=" * 60)
if 'Category' in df.columns and 'Subcategory' in df.columns:
    cat_sub = df.groupby('Category')['Subcategory'].nunique().sort_values(ascending=False)
    print("Unique Subcategories per Category:")
    print(cat_sub.to_string())

    # Top Category
    top_cat = df['Category'].value_counts().index[0]
    top_sub_in_cat = df[df['Category'] == top_cat]['Subcategory'].value_counts().head(15)
    print(f"\nTop Subcategories in '{top_cat}':")
    print(top_sub_in_cat.to_string())

# ── 10. Service → Subcategory relationship ───────────────────────────────────
print("\n" + "=" * 60)
print("SERVICE → SUBCATEGORY MAPPING")
print("=" * 60)
if 'Service' in df.columns and 'Subcategory' in df.columns:
    svc_sub = df.groupby('Service')['Subcategory'].nunique().sort_values(ascending=False)
    print("Unique Subcategories per Service (top 20):")
    print(svc_sub.head(20).to_string())

# ── 11. Duplicate analysis ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)
dup_incidents = df['IncidentNumber'].duplicated().sum()
print(f"Duplicate IncidentNumber: {dup_incidents}")
dup_all = df.duplicated().sum()
print(f"Fully duplicate rows: {dup_all}")

# ── 12. Recommendations ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PREPROCESSING RECOMMENDATIONS")
print("=" * 60)
recs = """
═══════════════════════════════════════════════════════════════════
  DATA CLEANING
═══════════════════════════════════════════════════════════════════
• Drop rows where Subcategory is null — target must be known.
• Drop duplicate IncidentNumber rows.
• CustomerLocation has high cardinality → group rare locations into 'Other'
  or use as-is depending on model (tree-based can handle it).
• Subject/Symptom: strip emails, phone numbers, URLs, boilerplate
  (signatures, disclaimers) — very noisy free text.
• IncidentNumber is a unique ID — drop for modeling.
• ProfileFullName may correlate with Subcategory (same person files
  same type) — could drop or hash to avoid overfitting.
• SCMTitle may contain job roles → useful categorical/text feature.

═══════════════════════════════════════════════════════════════════
  FEATURE ENGINEERING
═══════════════════════════════════════════════════════════════════
• Text → TF-IDF (Subject + Symptom combined). Limit to top 500–2000
  unigrams/bigrams. Symptom is longer → more signal.
• Category + Service + SCMTypeOfRequest → encode via target encoding
  or one-hot (cardinality is low-moderate).
• CustomerLocation → target encoding or frequency encoding.
• Create derived features:
    - Subject length (chars / words)
    - Symptom length (chars / words)
    - Whether Subject or Symptom contains specific keywords
    - SCMTitle → extract role keywords

═══════════════════════════════════════════════════════════════════
  TARGET HANDLING
═══════════════════════════════════════════════════════════════════
• Subcategory is highly imbalanced (top ~20 classes dominate).
• Option A: Predict only top-K (e.g., top 50) subcategories,
  group rest as 'Other'.
• Option B: Use stratified splitting for train/test.
• Consider hierarchical classification if Category→Subcategory
  structure is strict.

═══════════════════════════════════════════════════════════════════
  MODELING SUGGESTIONS
═══════════════════════════════════════════════════════════════════
• Baseline: Logistic Regression (with TF-IDF + OHE).
• Best bet: LightGBM or XGBoost — handles mixed text+tabular,
  categorical features, missing values natively.
• Text-only baseline: TF-IDF → Linear SVM or Naive Bayes.
• Evaluate with macro-F1 and weighted-F1 (not accuracy).
• Use 5-fold stratified cross-validation.

═══════════════════════════════════════════════════════════════════
  CORRELATION INSIGHTS (from Cramér's V)
═══════════════════════════════════════════════════════════════════
• Check the Cramér's V heatmap in analysis_output/.
• High V between Category and Subcategory → good predictor.
• High V between Service and Subcategory → good predictor.
• If Category/Service are nearly deterministic of Subcategory,
  a simple classifier may suffice.
═══════════════════════════════════════════════════════════════════
"""
print(recs)
with open(OUT_DIR / "preprocessing_recommendations.txt", 'w') as f:
    f.write(recs)

# ── 13. Summary report ───────────────────────────────────────────────────────
summary = f"""
================================================================
SUMMARY REPORT
================================================================
Total rows:          {len(df):>10,}
Total columns:       {len(df.columns):>10}
Target (Subcategory)
  Nulls:             {target_null:>10,} ({target_null/len(df)*100:.2f}%)
  Unique classes:    {len(vc):>10}
  Null rate (any col): {missing_df['Percent'].max():.2f}%

Columns by null rate:
{missing_df.to_string()}

Top-5 Subcategories:
{vc.head(5).to_string()}

Cramér's V matrix:
{corr_matrix.round(3).to_string()}
================================================================
"""
with open(OUT_DIR / "summary_report.txt", 'w') as f:
    f.write(summary)
print(summary)
