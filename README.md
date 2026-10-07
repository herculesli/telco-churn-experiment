![Churn rate by contract length](assets/header.png)
# Telco Churn Experiment: Churn Modeling + A/B Test Design

**End-to-end product DS portfolio project: from churn modeling to a pre-registered A/B test proposal for a high-risk telco subscriber segment.**

A logistic regression model identifies subscribers likely to churn, and a full experiment proposal — hypothesis, power analysis, decision framework, guardrails, and pre-registered segmentation — operationalizes the insight into a low-cost, high-scale retention intervention.

> Built to demonstrate how a data scientist scopes a problem end-to-end: business framing → modeling → causal experiment design → pre-registered decision criteria. Not "I trained a model and got 89% recall."

---

## TL;DR

| | |
|---|---|
| **Problem** | Month-to-month subscribers churn at 42% vs 2% for two-year contract holders — a 21x gap. ~55% of the customer base. |
| **Insight** | Churn is driven by **perceived value**, not price. Premium-priced users *with* bundled services retain better than premium-priced users *without*. Contract length and feature bundling are the dominant controllable levers. |
| **Action** | Offer high-risk users a discounted Security + Device Protection bundle, conditional on a 1-year contract upgrade. |
| **Validation** | Pre-registered A/B test. Primary metric: 30-day retention rate. MDE = 3pp absolute. |
| **Projected lift** | ~116 retained users / quarter / 3,875-user segment → ~$104K ARR preserved at current base size (assuming blended ARPU ~$75). |

[Read the full experiment proposal →](docs/Experiment_Proposal_Bundle_Retention.pdf)

---

## Why this project?


1. **Problem framing.** The model is a means, not the deliverable. The deliverable is a decision framework that tells the business what to ship.
2. **Tradeoff fluency.** Logistic regression over XGBoost is a deliberate explainability tradeoff. 89% recall at the cost of 43% precision is a deliberate intervention-cost tradeoff. Both are defended in writing.
3. **Experimentation rigor.** Hypothesis tied to the primary metric. Power analysis with explicit MDE. Pre-registered SHIP / ITERATE / KILL criteria. Guardrails on ARPU, support load, and unsubscribes. Inference protocol declared upfront to prevent peeking. Non-goals scope what the test does *not* cover.
4. **Causal thinking.** Bundle adopters self-select on engagement and intent. The proposal explains why randomization is the only way to isolate the bundle's causal lift, not just observe correlations.

---

## Repo Structure

```
.
├── README.md
├── data/
│   ├── Telco-Customer-Churn.csv     # Raw IBM Telco data
│   └── telco_churn_clean.csv        # Cleaned output of load_data.py
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory analysis, segment sizing
│   └── 02_analysis.ipynb            # Modeling, feature engineering, evaluation
├── src/
│   └── load_data.py                 # ETL: clean CSV → SQLite (telco_churn.db)
├── models/
│   └── m2m_churn_lr_model_v1.pkl    # Trained logistic regression (joblib-loadable)
├── docs/
│   └── Experiment_Proposal_Bundle_Retention.pdf  # Full A/B test proposal
├── telco_churn.db                   # SQLite store — queryable post-ETL
├── .gitignore
└── requirements.txt
```

**Why SQLite?** The ETL stages cleaned data into a SQLite database for queryable access from notebooks via SQL — closer to a production analytics workflow than `pd.read_csv()` everywhere. Lets the same dataset be hit with SQL in EDA and pandas in modeling without duplicating logic.

---

## Methodology

### Model

**Logistic Regression** with class weights `{0: 1, 1: 5}` to favor recall on the minority (churn) class.

Chosen over tree ensembles for **coefficient interpretability** — the model is the input to a stakeholder-facing recommendation, so reviewers need to see *why* a user was flagged. Next iteration would benchmark against XGBoost + SHAP for predictive lift while preserving explainability.

### Feature Engineering

| Feature | Rationale |
|---|---|
| `tenure` | Strong retention signal; survivorship proxy |
| `monthlycharges` | Price sensitivity proxy |
| `contract` | Single largest retention lever in the data |
| `online_security`, `device_protection` | Bundle attach signals |
| `payment_method` | Electronic check correlates with churn — likely a friction proxy |

`totalcharges` was **dropped** due to near-perfect collinearity with `tenure × monthlycharges`. Retaining it produced unstable coefficients in cross-validation.

### Performance

| Metric | Value | Notes |
|---|---|---|
| Recall (churn) | **89%** | Operative metric — drives intervention triggering |
| Precision (churn) | 43% | Bounds intervention cost ceiling |
| Accuracy | 66% | Intentionally below 73% no-information baseline; class weights trade accuracy for recall by design |

**Why this tradeoff is acceptable:** the intervention is software-only (in-app modal + email). Marginal cost per false positive ≈ $0. A flat cash discount would require precision ≥ 70% to be viable; a conditional bundle-with-contract-upgrade extracts guaranteed LTV from the 57% of false positives, mitigating the cost.

---

## Experiment Design

Full proposal in [`docs/Experiment_Proposal_Bundle_Retention.pdf`](docs/Experiment_Proposal_Bundle_Retention.pdf). Highlights:

- **Hypothesis.** Conditional bundle offer reduces 30-day churn rate in the high-risk segment by ≥3pp absolute, without material revenue loss.
- **Why randomized.** Bundle adopters self-select on engagement and intent — observational comparisons would conflate the bundle's causal effect with selection bias.
- **Randomization.** User-level, 50/50. No network effects → no clustering needed.
- **Power Analysis.** `n ≈ 16 × p(1−p) / MDE²` → 4,331 per arm, 8,662 total. α=0.05 two-tailed, 80% power. Baseline `p` is a placeholder; final value derived from production event logs pre-launch.
- **Decision Framework (pre-registered).** SHIP / ITERATE / KILL criteria declared *before* launch to prevent post-hoc rationalization.
- **Inference protocol.** Single primary read at week 8 to preserve nominal α = 0.05; weekly tracking is monitoring-only without stop authority.
- **Guardrails.** ARPU (−1%), gross revenue, support contact rate (+5%), unsubscribe rate (+2pp), organic bundle attach (cannibalization watch).
- **Pre-registered segmentation.** Tenure × monthly charge × payment method × service intensity. Subgroup p-values Bonferroni-corrected; segment findings treated as hypothesis-generating, not ship-gating.

---

## Reproducibility

```bash
# Setup
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# ETL: clean raw CSV → SQLite
python src/load_data.py

# Run analysis
jupyter lab notebooks/
# Open 01_eda.ipynb, then 02_analysis.ipynb
```

The trained model (`models/m2m_churn_lr_model_v1.pkl`) can be loaded directly:

```python
import joblib
model = joblib.load("models/m2m_churn_lr_model_v1.pkl")
```

---

## Tech Stack

`Python 3.11` · `pandas` · `scikit-learn` · `statsmodels` · `SQLite` · `Jupyter` · `matplotlib`

---

## Data

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,032 customers, 21 features, binary churn label.

Note: the public dataset is a static snapshot without an explicit time horizon attached to the churn label. The proposal uses this as a placeholder baseline and commits to re-deriving the production `p` from event logs pre-launch.

---

## What I'd do differently in v2

- **Survival analysis (Kaplan-Meier on `tenure`) to derive an empirical 30-day retention baseline** — the public dataset is a static snapshot with no time horizon attached to the churn label. v1 sidesteps this by deriving `p` operationally from production event logs pre-launch. v2 would fit a KM curve on `tenure × churn` to produce a data-grounded baseline, while accounting for left-truncation bias from the snapshot. This would also enable **CUPED-style variance reduction** in the experiment, cutting required `n` by ~30%.
- Benchmark XGBoost + SHAP against the logistic baseline; quantify the explainability/performance tradeoff in dollar terms.
- Add a propensity-score calibration arm to the experiment to address selection bias from the model's targeting.
- Sequential testing with alpha-spending (O'Brien-Fleming or Pocock) to enable safe interim reads instead of monitoring-only.

---

## Author

**Hercules Li** · MSBA, UCLA Anderson · [LinkedIn](https://www.linkedin.com/in/hercules-li-5a16a3192/) · [Email](mailto:lihercules97@gmail.com)
