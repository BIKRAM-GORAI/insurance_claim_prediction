# Project Progress & Contributor Tracker

This document tracks overall implementation progress, module ownership, and integration status for the Insurance Claim Prediction project. Team members should update their respective sections as they make progress or raise pull requests.

---

## 1. Overall Project Distribution (60% Backend / 40% Presentation)

| Workstream | Target Share | Lead / Contributor | Current Status | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Backend & Core ML Engine** | **60%** | Project Owner / Member 1 | **Completed & Hardened (100% of Phase 1)** | Data contract audit, deterministic split, model training pipeline, model serialization, Flask API routes, input validation & error handling, 38 passing API unit tests (including exhaustive edge cases), and comprehensive Jupyter notebook with Viva Q&A. |
| **Presentation & UI Layer** | **40%** | Frontend Contributor & Doc Lead | **Completed & Tested (100% of Phase 2)** | Responsive single-page web application (`templates/index.html`, `style.css`, `script.js`), frontend contract test suite (`tests/test_frontend_contract.py`), analytics/dataset-stats API endpoints, and setup documentation. |

---

## 2. Milestone Checklist

### Phase 1: 60% Backend & Core ML Engine (Completed & Hardened)
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
  - [x] Root route `GET /` serving responsive frontend `index.html`.
  - [x] Analytics API endpoints: `GET /api/metrics` and `GET /api/dataset-stats`.
- [x] **Automated Test Suite (`tests/`):**
  - [x] `tests/test_data_contract.py` (7 tests passing).
  - [x] `tests/test_training_pipeline.py` (5 tests passing).
  - [x] `tests/test_frontend_contract.py` (23 tests passing).
  - [x] `tests/test_api.py` (38 tests passing, comprehensive edge cases).
  - [x] Total: 73 / 73 tests passing.
- [x] **Jupyter Analysis Notebook (`notebooks/Insurance_Claim_Analysis.ipynb`):**
  - [x] Complete technical narrative across Sections 1 through 14 covering synthetic data governance, data audit, leakage prevention, ColumnTransformer & Pipeline, algorithmic comparison, and dynamic model selection.
  - [x] Dedicated Section 15: Comprehensive Viva Questions and Answers (13 questions across 6 core categories).

---

### Phase 2: 40% Presentation & Web UI Layer (Completed)
- [x] **Web Presentation Interface:**
  - [x] `templates/index.html`: Responsive single-page application with 4 views (Overview, Predict, Analytics, History), clear USD units, disclaimer.
  - [x] `static/css/style.css`: Modern, professional insurance dashboard styling with responsive design for desktop & mobile.
  - [x] `static/js/script.js`: Client-side validation, async API calls (`/api/predict`, `/api/metrics`, `/api/dataset-stats`), loading states, history management with CSV export.
- [x] **Contract & Integration Tests:**
  - [x] `tests/test_frontend_contract.py`: HTML structure, element IDs, allowed category levels, analytics endpoint contract, edge cases.
  - [x] Documentation & guides: `README_SETUP.md`, `DEMO_GUIDE.md`, and `PROJECT_AUDIT.md`.

---

## 3. Team Member Contributions Log

| Date | Contributor | Branch / PR | Changes Made | Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-27 | Project Owner | `main` | Initialized Git repository, remote origin, `.gitignore`, `requirements.txt`, data audit, `train_models.py` (with 4 models & dynamic selection), `app.py`, 21 unit tests, report outputs & figures, and `Insurance_Claim_Analysis.ipynb`. | **Completed (Phase 1 / 60% Backend)** |
| 2026-10-27 | Full-Stack Integrator | `feature/hackathon-improvements` | Built 4-tab SPA web frontend (`templates/index.html`, `static/css/style.css`, `static/js/script.js`), analytics endpoints (`/api/metrics`, `/api/dataset-stats`), frontend contract test suite (`tests/test_frontend_contract.py`), setup & demo guides. | **Completed (Phase 2 / 40% Frontend)** |
| 2026-10-27 | Member 1 | `feature/member1-backend-notebook` | **Backend Edge-Case Hardening & Notebook Viva Enrichment:** Extended `tests/test_api.py` with 29 new edge-case tests (forbidden fields like `claim_status` & `source_profile_id`, nulls, blanks, boolean-to-int prevention, NaN/infinity, fractional ints, bounds, boundary min/max, pipeline Approved class mapping, safe 500 error handling without leaking paths/traces); enhanced all 12 Markdown sections of `notebooks/Insurance_Claim_Analysis.ipynb` with detailed engineering concepts; added Section 15 "Comprehensive Viva Questions and Answers" (13 technical Q&As across 6 categories); updated `PROGRESS.md`. Full suite passing: 73/73 tests. | **Completed (Member 1 Scope)** |
