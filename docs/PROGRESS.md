# Project Progress & Contributor Tracker

This document tracks overall implementation progress, module ownership, and integration status for the Insurance Claim Prediction project. Team members should update their respective sections as they make progress or raise pull requests.

---

## 1. Overall Project Distribution (60% Backend / 40% Presentation)

| Workstream | Target Share | Lead / Contributor | Current Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Backend & Core ML Engine** | **60%** | Project Owner / Integrator | **Completed & Tested (100% of Phase 1)** | Data contract audit, deterministic split, model training pipeline, model serialization, Flask API routes, input validation & error handling, 21 passing automated tests, core Jupyter notebook. |
| **Presentation & UI Layer** | **40%** | Frontend Contributor & Doc Lead | **Completed & Tested (100% of Phase 2)** | Web presentation interface (`templates/index.html`, `style.css`, `script.js`), frontend contract test (`tests/test_frontend_contract.py`), comprehensive viva QA guide (`PROJECT_WORKFLOW_AND_VIVA_QA.md`). |

---

## 2. Milestone Checklist

### Phase 1: 60% Backend & Core ML Engine (Completed)
- [x] **Git Repository Setup:**
  - [x] Initialized Git repository on `main` branch.
  - [x] Configured remote origin: `https://github.com/BIKRAM-GORAI/insurance_claim_prediction`.
  - [x] Configured `.gitignore` and `requirements.txt`.
  - [x] Structured `data/` directory with `insurance_claim_dataset_cleaned.csv`.
  - [x] Provided top-level `README.md`.
- [x] **Dataset Contract & Quality Audit:**
  - [x] Verified dataset shape: 5,000 rows × 40 columns.
  - [x] Verified target `claim_status` distribution: Approved: 3,042 (60.84%), Rejected: 1,958 (39.16%).
  - [x] Checked 0 nulls across all features.
  - [x] Verified 9 model feature bounds and exact category levels.
  - [x] Verified total claim amount equals sum of components (`injury_claim + property_claim + vehicle_claim`).
- [x] **Model Training & Evaluation Pipeline (`train_models.py`):**
  - [x] Deterministic 80/20 grouped split using `source_profile_id` (seed `42`, 792 train groups, 197 holdout groups).
  - [x] Scikit-learn preprocessing `ColumnTransformer` (StandardScaler for numerics, OneHotEncoder for categoricals).
  - [x] 5-fold `GroupKFold` cross-validation on train partition for 4 candidate models:
    - Logistic Regression (CV Acc: 0.8414 ± 0.0225, Holdout Acc: 0.8995, Macro F1: 0.8886, ROC-AUC: 0.9253)
    - Decision Tree (CV Acc: 0.8285 ± 0.0213, Holdout Acc: 0.8482, Macro F1: 0.8407, ROC-AUC: 0.8830)
    - Random Forest (CV Acc: 0.8607 ± 0.0265, Holdout Acc: 0.8744, Macro F1: 0.8603, ROC-AUC: 0.9336)
    - Gradient Boosting (CV Acc: 0.8599 ± 0.0181, Holdout Acc: 0.8975, Macro F1: 0.8862, ROC-AUC: 0.9439)
  - [x] Dynamic model selection: selects top model based on holdout accuracy (selected: `LogisticRegression` at 89.95%).
  - [x] Serialized winning pipeline to `models/claim_model_pipeline.joblib`.
  - [x] Model metadata saved to `models/model_metadata.json`.
  - [x] Model comparison exported to `reports/model_comparison.json`.
  - [x] Figures generated to `reports/figures/` (confusion matrices & comparison bar chart).
- [x] **Flask API (`app.py`):**
  - [x] `GET /api/health` endpoint with model status and loaded model name.
  - [x] `POST /api/predict` endpoint with strict 9-field schema validation, bounds checks, category validation, whole integer enforcement.
  - [x] Dynamic lookup for class `Approved` score calculation from `pipeline.classes_`.
  - [x] Safe error response format (`ValidationError`, HTTP 400, HTTP 500 without leaking stack traces or local paths).
  - [x] Root route `GET /` serving API description or `index.html`.
- [x] **Automated Backend Test Suite (`tests/`):**
  - [x] `tests/test_data_contract.py` (7 tests passing).
  - [x] `tests/test_training_pipeline.py` (5 tests passing).
  - [x] `tests/test_api.py` (9 tests passing).
  - [x] Total: 21 / 21 tests passing.
- [x] **Jupyter Analysis Notebook (`notebooks/Insurance_Claim_Analysis.ipynb`):**
  - [x] Covers Sections 1 through 14 of the project specification (EDA, data quality audit, grouped splitting, pipeline construction, model comparison, dynamic selection, artifact reload, and sample prediction).

---

### Phase 2: 40% Presentation & Viva Layer (Completed)
- [x] **Web Presentation Interface (Person 3):**
  - [x] `templates/index.html`: Responsive form containing the 9 inputs in documented order, clear USD units, disclaimer, quick presets.
  - [x] `static/css/style.css`: Clean, modern styling, card layout, glassmorphic design, responsive design for desktop & mobile.
  - [x] `static/js/script.js`: Client-side validation, async `fetch('/api/predict')`, loading states, error presentation, form reset.
- [x] **Contract & Integration Tests (Person 4):**
  - [x] `tests/test_frontend_contract.py`: HTML element IDs and dropdown category value assertions (8 passing tests).
  - [x] Total: 29 / 29 automated tests passing across the system.
  - [x] Presentation slides & viva talking points outline (documented in `PROJECT_WORKFLOW_AND_VIVA_QA.md`).

---

## 3. Team Member Contributions Log

| Date | Contributor | Branch / PR | Changes Made | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-27 | Project Owner | `main` | Initialized Git repository, remote origin, `.gitignore`, `requirements.txt`, data audit, `train_models.py` (with 4 models & dynamic selection), `app.py`, 21 unit tests, report outputs & figures, and `Insurance_Claim_Analysis.ipynb`. | **Completed (Phase 1 / 60% Backend)** |
| 2026-10-29 | Frontend Contributor | `feature/frontend` | Implemented `templates/index.html`, `static/css/style.css`, `static/js/script.js`, and `tests/test_frontend_contract.py` (8 contract tests passing). | **Completed (Phase 2 Presentation)** |

