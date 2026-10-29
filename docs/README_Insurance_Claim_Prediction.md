# Insurance Claim Prediction Using Machine Learning

**Project specification and team contract --- Version 1.0**

> This README is the single source of truth for the project. All team
> members, notebooks, Python modules, API responses, tests, report
> figures, and presentation slides must follow it. Do not independently
> rename fields, invent categories, change model inputs, alter the
> target, or change the evaluation split. If a change is necessary,
> propose it, update this README, and update the tests in the same pull
> request.

------------------------------------------------------------------------

## 1. Project goal and scope

Build a small web application that accepts nine claim-related inputs and
predicts the `claim_status` label using supervised machine learning.

The project has two connected deliverables:

1.  **ML analysis notebook:** explains the dataset, cleaning,
    exploratory data analysis (EDA), model comparison, evaluation, and
    model selection. This is the main artifact for demonstrating the
    work in the report and viva.
2.  **Flask web application:** accepts the same nine fields, loads the
    saved preprocessing-and-model pipeline, and returns a prediction in
    a fixed JSON format.

### Important limitation

The dataset's `claim_status` labels are **synthetically generated** to
represent a simulated underwriting policy. This project predicts that
synthetic label. It must not claim to predict actual insurance-company
decisions or real-world claim outcomes.

Use language such as: - "Predicted claim status under the synthetic
dataset policy" - "Model-estimated approval score" - "Holdout
performance on this dataset"

Do not use language such as: - "Guaranteed approval" - "The insurer will
approve this claim" - "89% real-world approval probability"

The displayed score is a model output. Unless calibration has been
separately evaluated, do not describe it as a calibrated real-world
probability.

### Out of scope for version 1.0

-   User accounts, authentication, database storage, or claim history
-   Payment systems, external insurer APIs, or real insurance
    integrations
-   React, Tailwind CSS, or a JavaScript backend
-   Complex explainability frameworks
-   Live retraining from the web form
-   Adding or removing model features without approval
-   Altering labels or injecting noise to achieve a desired score

------------------------------------------------------------------------

## 2. Fixed technology choices

  -----------------------------------------------------------------------
  Area                                Required choice
  ----------------------------------- -----------------------------------
  Language                            Python 3.11 (use the same minor
                                      version for development where
                                      possible)

  Data analysis                       Jupyter Notebook

  Data processing / ML                pandas, NumPy, scikit-learn

  Charts                              Matplotlib and Seaborn

  Backend                             Flask

  Frontend                            HTML, CSS, vanilla JavaScript

  Model artifact                      One serialized scikit-learn
                                      `Pipeline` for the selected model

  Tests                               pytest or Python `unittest`; use
                                      one consistently

  Version control                     Git and GitHub, feature branches
                                      and pull requests

  Environment                         Python virtual environment and
                                      `requirements.txt`
  -----------------------------------------------------------------------

Do not use React or Tailwind CSS. The frontend must call the Flask API;
it must not reproduce the model logic in JavaScript.

### Notebook versus Python files

-   The notebook is for analysis, charts, explanations, and the
    model-comparison narrative.
-   `train_models.py` is the reproducible training/evaluation entry
    point.
-   `app.py` is the Flask application and prediction API.
-   The notebook and script must use the same target, nine features,
    category mappings, preprocessing approach, and grouped-split policy.
-   Do not maintain two different copies of preprocessing logic. The
    saved pipeline must contain preprocessing and the classifier
    together.

------------------------------------------------------------------------

## 3. Dataset contract

### 3.1 Input file

Expected input file:

`data/insurance_claim_dataset_cleaned.csv`

The cleaned dataset currently has **5,000 rows and 40 columns**,
including: - 9 model input features - 1 target column (`claim_status`) -
1 lineage/grouping column (`source_profile_id`) - additional columns
retained for analysis or excluded from modelling

The reported target distribution is:

  Target          Rows    Share
  ------------ ------- --------
  `Approved`     3,042   60.84%
  `Rejected`     1,958   39.16%

At training time, print the actual shape, target counts, unique
source-profile count, and missing-value summary. If the file differs
from these expected values, stop and investigate rather than silently
continuing.

### 3.2 Target

-   Target column: `claim_status`
-   Allowed labels: `Approved`, `Rejected`
-   The target is `y`; it is never an input feature.
-   Do not regenerate, rewrite, rebalance, or manually edit target
    labels during model development.
-   Do not remove rows just because a model predicts them incorrectly.
-   Any approved change to the label-generation policy must be
    documented and trigger a full reevaluation.

### 3.3 Source-profile grouping field

-   Exact column: `source_profile_id`
-   Purpose: identifies the original source profile associated with a
    row in the expanded dataset.
-   It is **only for splitting/grouping and audit checks**.
-   It must never be included in `X`, one-hot encoded, scaled, sent from
    the frontend, or accepted by the prediction API.
-   Rows with the same `source_profile_id` must stay entirely in the
    same split. This prevents related/bootstrapped rows from appearing
    in both training and holdout data.

### 3.4 Exact model features

These nine fields are the only permitted model inputs in version 1.0.
Keep the spelling and case exactly as shown.

  ------------------------------------------------------------------------------------------
                  Order Field                           Type             Accepted UI/API
                                                                         values
  --------------------- ------------------------------- ---------------- -------------------
                      1 `incident_severity`             categorical      `Trivial Damage`,
                                                        string           `Minor Damage`,
                                                                         `Major Damage`,
                                                                         `Total Loss`

                      2 `police_report_available`       categorical      `YES`, `NO`,
                                                        string           `Unknown`

                      3 `authorities_contacted`         categorical      `Police`, `Fire`,
                                                        string           `Ambulance`,
                                                                         `Other`,
                                                                         `None Contacted`

                      4 `witnesses`                     integer          0--3 inclusive

                      5 `total_claim_amount`            numeric          USD; observed
                                                                         dataset range
                                                                         approximately
                                                                         98--116602

                      6 `policy_deductable`             integer          500, 1000, 2000

                      7 `months_as_customer`            integer          0--479 inclusive

                      8 `number_of_vehicles_involved`   integer          1--4 inclusive

                      9 `incident_hour_of_the_day`      integer          0--23 inclusive
  ------------------------------------------------------------------------------------------

For `total_claim_amount`, the UI may accept decimal USD amounts, but it
must reject non-finite values, zero/negative amounts, and amounts
outside the documented version-1.0 input bounds. Use the observed
dataset minimum and maximum as the initial bounds; confirm the exact
minimum/maximum from the CSV in the training validation code and record
them in the notebook. Do not silently clip or alter a user's amount.

**Important:** The accepted bounds above describe this project's
supported input range. They do not imply that other values are
impossible in real life. Inputs outside the supported range must receive
a validation error, not a fabricated prediction.

### 3.5 Category normalization and missing values

The cleaned dataset uses explicit categories rather than the literal `?`
marker. The intended meanings are:

-   `Unknown`: the information is not known or is unavailable.
-   `None Contacted`: no authority/first responder was contacted,
    according to the dataset's coding.
-   `NO`: an explicit negative answer; it is not interchangeable with
    `Unknown`.
-   `YES`: an explicit affirmative answer.

Rules: 1. Use the exact category strings in this README everywhere. 2.
Do not substitute `None`, `None Contacted`, `Unknown`, `NO`, and `False`
for one another. 3. Do not use empty strings or `NaN` as frontend
category values. 4. The backend must reject unrecognized category
strings with HTTP 400. 5. `OneHotEncoder(handle_unknown="ignore")` is a
defensive implementation setting, not permission for the UI/API to
accept arbitrary category values. 6. Check that the cleaned CSV actually
contains the documented category values before training. If it does not,
stop and update the specification deliberately.

### 3.6 Columns not allowed in the model

Only the nine fields in section 3.4 may be in `X`. Exclude all other
columns, including: - `claim_status` from `X` (it is `y`) -
`source_profile_id` (group splitting only) - `policy_number` -
`insured_zip` - `incident_location` - raw date fields such as
`policy_bind_date` and `incident_date` - `fraud_reported` if it appears
in an original/source copy - the individual claim components
`injury_claim`, `property_claim`, and `vehicle_claim` - any
target-generation helper, source-row index, or generated ID - every
other column not explicitly listed among the nine features

The individual claim components sum to `total_claim_amount` in the
cleaned dataset and are redundant for this version. Do not add them as
extra inputs.

The training script must explicitly select the nine features; it must
not use "all columns except target" as a shortcut.

------------------------------------------------------------------------

## 4. Required repository structure

Use this structure unless the integrator approves a documented change.

``` text
insurance-claim-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── train_models.py
├── data/
│   └── insurance_claim_dataset_cleaned.csv
├── notebooks/
│   └── Insurance_Claim_Analysis.ipynb
├── models/
│   ├── claim_model_pipeline.joblib
│   └── model_metadata.json
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── tests/
│   ├── test_data_contract.py
│   ├── test_training_pipeline.py
│   ├── test_api.py
│   └── test_frontend_contract.py
└── reports/
    ├── figures/
    └── model_comparison.json
```

### File ownership

-   `notebooks/Insurance_Claim_Analysis.ipynb`: project owner / ML owner
-   `train_models.py`: project owner / ML owner
-   `models/`: generated by the training script; do not hand-edit
-   `app.py`: backend contributor, integrated by project owner
-   `templates/index.html`, `static/css/style.css`,
    `static/js/script.js`: frontend contributor
-   `tests/`: each contributor adds tests for their module; integrator
    owns full integration tests
-   `README.md`, `requirements.txt`, `.gitignore`: integrator/project
    owner reviews all changes

Do not create alternate versions such as `app_final2.py`,
`model_new.pkl`, or `script_latest.js`. Use Git history and pull
requests instead.

### Generated files

-   `models/claim_model_pipeline.joblib` must be generated by
    `train_models.py`.
-   `reports/model_comparison.json` must be generated from real
    evaluation results.
-   Charts must be generated by the notebook or scripts and saved to
    `reports/figures/`.
-   Never type model metrics manually into generated result files.
-   Do not commit virtual environments, Python cache folders, editor
    settings containing personal paths, or secrets.

------------------------------------------------------------------------

## 5. Data splitting and evaluation rules

### 5.1 Holdout split

Use a deterministic grouped holdout split based on `source_profile_id`.

Required behaviour: 1. Get the unique `source_profile_id` values. 2.
Sort the group IDs before splitting so input row order does not change
the result. 3. Use a fixed seed of `42`. 4. Assign whole source profiles
to training or holdout; never split rows from the same profile across
both sets. 5. Use an approximately 80/20 group split. 6. Print and save
the number of rows and unique groups in each partition. 7. Assert that
the intersection of train and holdout group IDs is empty. 8. Save the
actual split details to `reports/model_comparison.json` or associated
metadata.

The previous audit reported 792 training profiles and 197 holdout
profiles, with 4,005 training rows and 995 holdout rows. These counts
are reference results from that audit, **not a reason to force the
split**. The code must use the deterministic procedure above and report
its actual counts. If the new implementation yields different counts,
investigate the split implementation and document the result; do not
manually move rows to make the counts match.

A recommended implementation is to split the sorted unique group IDs
with a fixed seed and a fixed holdout group count of 197, then filter
the rows by membership. This gives 792 training groups and 197 holdout
groups for 989 total profiles. Record the exact method in code and
metadata.

### 5.2 Cross-validation

-   Use `GroupKFold(n_splits=5)` on the development/training partition
    only.
-   Pass `source_profile_id` as the grouping variable.
-   The holdout set must not participate in cross-validation, model
    selection, preprocessing fitting, or threshold tuning.
-   Any preprocessing such as scaling and one-hot encoding must be
    inside a scikit-learn `Pipeline` so it is fitted separately within
    each training fold.
-   Report mean and standard deviation of cross-validation accuracy.
-   Do not use ordinary random row-wise cross-validation.

### 5.3 Evaluation metrics

Required, in this order: 1. Holdout accuracy 2. Confusion matrix 3.
Classification report: precision, recall, F1-score, support 4. Balanced
accuracy 5. ROC-AUC, when the model supports probability/score output
and it is computed correctly

The notebook must explain metrics in plain language. Accuracy is the
share of correct predictions. The confusion matrix shows which
`Rejected` cases were predicted as `Approved` and which `Approved` cases
were predicted as `Rejected`.

Because the class distribution is not exactly balanced, do not use
accuracy alone to justify the model. Consider class-level
precision/recall and the confusion matrix. Do not claim that one metric
proves real-world insurance reliability.

### 5.4 No leakage and no metric manipulation

-   Never split individual rows randomly while ignoring
    `source_profile_id`.
-   Never include `source_profile_id` as a feature.
-   Never fit a scaler or encoder on the holdout data.
-   Never choose the model based on holdout metrics and then repeatedly
    tune against that same holdout while continuing to present it as an
    untouched final test.
-   Never alter target labels, inject random noise, remove difficult
    examples, or change the split to obtain a preferred accuracy.
-   All models must be evaluated on exactly the same holdout rows.
-   Keep the split seed, package versions, feature list, and results in
    the report.
-   Report actual results, including disappointing results.

------------------------------------------------------------------------

## 6. Models and preprocessing

### 6.1 Required comparison models

Train and compare these four standard classifiers:

1.  Logistic Regression
2.  Decision Tree
3.  Random Forest
4.  Gradient Boosting

Use sensible, documented parameters. Avoid large hyperparameter searches
due to the short project timeline. Use fixed random seeds for models
that support them.

Initial comparison defaults: - Logistic Regression: `max_iter=2000`,
`random_state=42` where supported. - Decision Tree: `max_depth=5`,
`random_state=42`. - Random Forest: `n_estimators=100`,
`random_state=42`, `n_jobs=-1`. - Gradient Boosting: `random_state=42`.

These are comparison defaults, not a guarantee that a model will win. Do
not silently change them between the notebook and training script.

### 6.2 Preprocessing

Numeric features: - `witnesses` - `total_claim_amount` -
`policy_deductable` - `months_as_customer` -
`number_of_vehicles_involved` - `incident_hour_of_the_day`

Categorical features: - `incident_severity` -
`police_report_available` - `authorities_contacted`

Use a `ColumnTransformer`: - Numeric branch: `StandardScaler` -
Categorical branch: `OneHotEncoder(handle_unknown="ignore")`

Place the `ColumnTransformer` and classifier together inside one
scikit-learn `Pipeline`. All models should use the same feature set and
preprocessing strategy unless a documented technical reason requires a
difference.

### 6.3 Selecting and saving the final model

-   Compare all four models using grouped CV and the same grouped
    holdout set.
-   Use holdout accuracy, balanced accuracy, macro F1, and the confusion
    matrix together to make a reasoned choice.
-   Logistic Regression is the initial candidate because the prior audit
    reported the highest holdout accuracy; it is not automatically the
    winner in a new run.
-   Record the selected model name and selection rationale in the
    notebook, README results section, and `models/model_metadata.json`.
-   Save the complete selected pipeline to
    `models/claim_model_pipeline.joblib`.
-   The backend loads this single artifact. It must not separately load
    a scaler or encoder.
-   If the final model is retrained on all available labeled rows after
    evaluation, preserve the untouched holdout results from the
    pre-retraining evaluation and clearly distinguish the final
    deployment artifact from the evaluated model. Do not evaluate an
    all-data-fitted model on the same data and report it as holdout
    performance.

### 6.4 Model probabilities

If the selected classifier provides `predict_proba`, compute the score
for the `Approved` class by looking up the corresponding label in
`pipeline.classes_`; do not assume a fixed probability-array position.
Return the model-estimated score for `Approved` regardless of the
predicted class.

Do not call this a calibrated real-world probability. No calibration
claim is allowed unless separately tested and documented.

------------------------------------------------------------------------

## 7. Notebook order and required presentation flow

The notebook must read like a clear report, from understanding the data
to selecting a model. Keep Markdown explanations before or after
important code/plot cells. Avoid unrelated experiments and duplicated
plots.

### Section 1 --- Title, objective, and scope

Include: - Project title - One-paragraph objective - What the model
predicts - Explicit note that the target is synthetic - Brief workflow
diagram or numbered workflow

### Section 2 --- Imports and reproducibility

-   Import libraries in one cell.
-   Set and document random seed `42`.
-   Print Python and key library versions.
-   Define dataset path and constants.
-   Do not scatter inconsistent seeds throughout the notebook.

### Section 3 --- Load and inspect the dataset

Required sequence: 1. Load the CSV. 2. Display `df.shape`. 3. Display
the first five rows. 4. Display the last five rows only if useful. 5.
Display column names and data types. 6. Display `df.info()` output. 7.
Display `df.describe()` for numeric columns. 8. Display target counts
and percentages. 9. Display the number of unique `source_profile_id`
groups.

Explain what each inspection step tells the reader.

### Section 4 --- Data-quality audit

Required: - Missing-value count by column - Count of literal `?` values,
if any - Duplicate-row count - Duplicate or repeated `policy_number`
count as a data-quality observation, not a feature - Allowed-category
checks for the nine features - Target-label validation -
`source_profile_id` group-size summary - Verify `total_claim_amount`
against the sum of claim components where all components exist

Do not silently mutate data in this section. Show the audit first, then
describe the normalization already present in the cleaned CSV. If
additional cleaning is needed, implement it in one explicit,
reproducible preprocessing function and test it.

### Section 5 --- Explain features and exclusions

-   Show a table of the nine selected features with meanings and types.
-   Explain why `claim_status` is the target.
-   Explain why `source_profile_id` is only for grouped splitting.
-   Explain why IDs, raw dates, `fraud_reported`, and individual claim
    components are excluded.
-   Show the exact `X` and `y` construction in code.

### Section 6 --- Exploratory data analysis (EDA)

Create these plots in this order: 1. Target count plot: Approved vs
Rejected. 2. Histograms for `total_claim_amount` and
`months_as_customer`. 3. Count plots for `incident_severity`,
`police_report_available`, and `authorities_contacted`. 4. Boxplot of
`total_claim_amount` by `claim_status`. 5. Claim-status distribution by
`incident_severity`. 6. Claim-status distribution by
`police_report_available`. 7. Claim-status distribution by
`authorities_contacted`. 8. Numeric correlation heatmap for the six
numeric model features.

For each graph, add a short Markdown interpretation. State only what the
plot supports; do not infer causation from correlation. Clearly label
USD values as USD. A correlation heatmap is descriptive and does not
justify adding excluded fields to the model.

### Section 7 --- Grouped split

-   Explain why row-wise splitting is unsafe for bootstrapped/related
    profiles.
-   Create the deterministic grouped holdout split.
-   Print row counts and unique group counts for both partitions.
-   Assert no overlap of group IDs.
-   Show target distribution for training and holdout.
-   Keep the holdout data out of later fitting and cross-validation.

### Section 8 --- Preprocessing pipeline

-   Declare numeric and categorical feature lists.
-   Build the `ColumnTransformer`.
-   Explain scaling and one-hot encoding in plain language.
-   Put preprocessing inside the model pipeline.
-   Verify transformed data can be produced without leakage.

### Section 9 --- Train and compare four models

-   Train Logistic Regression, Decision Tree, Random Forest, and
    Gradient Boosting.
-   Use 5-fold GroupKFold on the training partition.
-   Show a comparison table with mean CV accuracy ± standard deviation.
-   Fit each candidate on the complete training partition and evaluate
    once on the same holdout partition.
-   Record results in a DataFrame and save them to
    `reports/model_comparison.json` or CSV.

### Section 10 --- Evaluation per model

For each of the four models, show: 1. Accuracy and balanced accuracy. 2.
Confusion matrix heatmap, with class labels ordered `Rejected`,
`Approved`. 3. Classification report. 4. ROC curve/AUC if correctly
available. 5. A brief, consistent interpretation.

Do not use different holdout sets for different models.

### Section 11 --- Model comparison and selection

Required: - Bar chart comparing holdout accuracy across the four
models. - Bar chart comparing holdout macro F1 across the four models. -
One comparison table with CV accuracy mean/std, holdout accuracy,
balanced accuracy, macro F1, and ROC-AUC where available. - Explain why
the selected model is appropriate using multiple metrics and error
types. - State the model-selection decision and its limitations. - Do
not state that the most complex model is automatically best.

### Section 12 --- Save and reload the final pipeline

-   Save the selected complete pipeline to
    `models/claim_model_pipeline.joblib`.
-   Save metadata: selected model, feature list, category mapping, seed,
    split method, dataset row count, package versions, and evaluation
    metrics.
-   Load the saved pipeline back from disk in a fresh cell/process.
-   Verify a sample input can be predicted after reloading.
-   Never save a pipeline before the final selection decision is
    recorded.

### Section 13 --- Sample prediction

-   Construct a sample as a DataFrame with the exact nine feature
    columns.
-   Show the predicted class and model-estimated approval score.
-   Explain that the example is illustrative and not a real insurer
    decision.
-   Do not hardcode an expected result unless the test is intentionally
    asserting a saved deterministic fixture.

### Section 14 --- Conclusion and limitations

Summarize: - What was done - Which model was selected and why - Actual
holdout results - Most important error patterns - Synthetic-label
limitation - Grouped-split approach - What the application can and
cannot claim

### Notebook quality rules

-   Use numbered Markdown headings matching the sequence above.
-   Use consistent chart titles, axis labels, class ordering, and figure
    sizes.
-   Put an explanation next to each major plot.
-   Do not include unrelated plots just to increase notebook length.
-   Run all cells from top to bottom before committing.
-   Remove stale outputs after changing the code, then rerun the entire
    notebook.
-   Do not leave exceptions, undefined variables, or results from a
    different dataset version.
-   Do not manually type metrics that should be calculated from model
    outputs.

------------------------------------------------------------------------

## 8. Flask API contract

### 8.1 Endpoint

`POST /api/predict`

Request content type: `application/json`

The endpoint accepts **exactly the nine model input fields**.
`source_profile_id`, `claim_status`, and other dataset columns are
forbidden in the request.

### 8.2 Request example

``` json
{
  "incident_severity": "Major Damage",
  "police_report_available": "YES",
  "authorities_contacted": "Police",
  "witnesses": 2,
  "total_claim_amount": 45000,
  "policy_deductable": 1000,
  "months_as_customer": 120,
  "number_of_vehicles_involved": 2,
  "incident_hour_of_the_day": 14
}
```

### 8.3 Successful response

``` json
{
  "status": "success",
  "claim_status": "Approved",
  "approval_score": 0.819,
  "model_name": "LogisticRegression"
}
```

The number `0.819` is an illustrative schema example, not a guaranteed
result for the request above. The implementation must return the actual
score produced by the loaded model.

`approval_score`: - Must be a JSON number from 0 to 1. - Represents the
selected model's score/probability output for class `Approved`. - Must
be present whether the predicted class is `Approved` or `Rejected`. -
Must not be rounded internally before the response is constructed; the
UI may display it as a percentage rounded to one decimal place.

### 8.4 Error response

Use HTTP 400 for invalid/missing input, with a stable structure:

``` json
{
  "status": "error",
  "error": "ValidationError",
  "message": "Invalid 'incident_severity'. Allowed values: Trivial Damage, Minor Damage, Major Damage, Total Loss."
}
```

Use HTTP 500 for unexpected server errors. Do not expose stack traces or
local filesystem paths to the browser.

### 8.5 Validation rules

-   All nine required keys must be present.
-   Reject nulls, blank strings, invalid types, and unknown categories.
-   Numeric fields must be numeric and finite; reject `NaN` and
    infinity.
-   Integer fields must be whole numbers, not silently truncated
    decimals.
-   Validate all numeric bounds from section 3.4.
-   `policy_deductable` must be one of 500, 1000, 2000.
-   `incident_hour_of_the_day` must be an integer from 0 to 23.
-   Return clear error messages.
-   Never silently fill missing values with a convenient default.
-   Do not trust client-side validation alone; validate again in Flask.

### 8.6 Health endpoint

Implement `GET /api/health`, returning a small JSON response that
confirms the API process is running and whether the model artifact
loaded successfully. Do not include sensitive filesystem information.

### 8.7 Frontend integration rules

-   The frontend must call the relative URL `/api/predict`.
-   Do not hardcode a separate localhost URL into `script.js`.
-   Use `fetch()` with `Content-Type: application/json`.
-   Disable the submit button while a request is pending and restore it
    afterward.
-   Display loading, success, validation-error, and unexpected-error
    states.
-   Never display stale results after a failed new request.
-   Show the predicted status and "Model-estimated approval score".
-   State clearly on the page that this is a student project using
    synthetic labels.
-   Escape or safely render API error text; do not inject untrusted
    strings as HTML.
-   Do not calculate the prediction or approval score in JavaScript.

------------------------------------------------------------------------

## 9. Frontend requirements

The frontend should be professional, responsive, and easy to demo. Avoid
unnecessary pages or features.

### Required UI

-   Project title and one-sentence description
-   Short synthetic-data disclaimer
-   Form with the nine fields in the exact schema order
-   Clear units for claim amount (USD)
-   Input constraints and understandable validation messages
-   Predict button and loading state
-   Result section showing predicted status and model-estimated approval
    score
-   Error display
-   Reset/clear form action
-   Responsive layout for laptop and mobile widths

### UI rules

-   Field names in the HTML may be human-friendly labels, but JSON keys
    must exactly match the API schema.
-   Use the exact category values from section 3.4.
-   Do not replace `Unknown` with a blank choice or reinterpret it as
    `NO`.
-   Use `None Contacted`, not `None`, for the authorities dropdown.
-   Do not show fabricated confidence, explanations, feature importance,
    or model metrics as if they came from a live computation.
-   If a project metrics section is added, load values from the saved
    report or clearly label them as training/evaluation results rather
    than the current prediction.

------------------------------------------------------------------------

## 10. Team roles and file boundaries

There are four people. The project owner performs most of the ML work,
much of the backend/structure work, integration, final testing, and
small fixes. Three other people receive bounded tasks.

### Person 1 --- Project owner / ML / integration

Responsible for: - Dataset contract and cleaning validation - Notebook
and EDA - Four-model comparison - Grouped split and evaluation -
`train_models.py` and selected model artifact - Main folder structure
and README - Reviewing API/schema consistency - Final integration, full
testing, and release

### Person 2 --- Backend contributor

Responsible for: - `app.py` - Flask route and validation
implementation - Loading the saved pipeline - API responses and error
handling - API tests

Constraints: - Must use the exact nine fields and categories from this
README. - Must not change model features or retrain the model
independently. - Must not rename API response keys. - Must not add
database or authentication features.

### Person 3 --- Frontend contributor

Responsible for: - `templates/index.html` - `static/css/style.css` -
`static/js/script.js` - Form UI, validation feedback, loading, result,
and error states

Constraints: - Must use the API contract exactly. - Must not calculate
ML predictions in JavaScript. - Must not add fields that the backend
does not accept. - Must not rename JSON keys or categories. - Can work
against the example API contract before the real backend is ready.

### Person 4 --- Small independent task / documentation support

Assign one bounded task, such as: - Report/PPT outline and captions for
generated figures, or - API test cases and frontend/backend contract
checks, or - README/setup verification and demo checklist.

Do not assign a vague task like "help with everything." One person owns
each file at a time.

### Integration owner

The project owner is the final integration owner. The frontend
contributor owns frontend implementation, but only the integrator merges
cross-module changes into the main branch.

------------------------------------------------------------------------

## 11. Git and collaboration rules

1.  `main` must always contain a runnable or clearly buildable project
    state.
2.  Each contributor creates a separate branch:
    -   `feature/ml-notebook`
    -   `feature/training-pipeline`
    -   `feature/flask-api`
    -   `feature/frontend`
    -   `docs/report-and-tests`
3.  Make small commits with descriptive messages.
4.  Push branches and open pull requests.
5.  The integrator reviews the diff and checks the relevant tests before
    merging.
6.  Never force-push to `main`.
7.  Do not commit generated secrets, `.env`, virtual environments,
    caches, or unrelated files.
8.  If changing the API schema, feature list, categories, file
    structure, split method, or model artifact name, update this README
    and relevant tests in the same PR.
9.  Do not rename a shared field "for readability." Human-readable
    labels belong in the UI; the API/model keys stay fixed.
10. Avoid concurrent edits to the same file. Agree on file ownership
    before coding.
11. After a merge, the integrator runs the relevant tests and verifies
    that the project still starts.
12. If a conflict affects shared contracts, the integrator resolves it
    according to this README rather than choosing one version
    arbitrarily.

------------------------------------------------------------------------

## 12. Dependencies and reproducibility

-   Create a virtual environment.
-   Install dependencies from `requirements.txt`.
-   Keep dependencies limited to those needed for analysis, ML, Flask,
    plotting, and testing.
-   Record the actual Python and library versions used for the final
    evaluation.
-   Use seed `42` consistently for split/model randomness where
    applicable.
-   Use deterministic group sorting before the holdout split.
-   The saved model should be regenerated with `train_models.py`, not
    manually edited.
-   `README.md` must contain exact setup, training, and run commands
    after the repository is created.
-   Do not claim perfect bitwise reproducibility across different
    operating systems/library versions; the goal is reproducible
    methodology and consistent schema/results within the documented
    environment.

Suggested setup commands (Windows PowerShell):

``` powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

After files are in place, the expected workflow is:

``` powershell
python train_models.py
python app.py
```

The notebook can be opened using:

``` powershell
jupyter notebook
```

or through the installed Jupyter support in VS Code. Exact commands may
be adjusted to the environment, but the README must document the final
tested commands.

------------------------------------------------------------------------

## 13. Required tests and release gate

Do not call the project complete until these checks pass.

### Data and training

-   [ ] CSV loads and has the expected columns.
-   [ ] Target labels are only `Approved` and `Rejected`.
-   [ ] All nine features exist and have valid values.
-   [ ] `source_profile_id` is never included in model features.
-   [ ] No source group overlaps between train and holdout.
-   [ ] Cross-validation passes groups correctly.
-   [ ] All four models use the same holdout rows.
-   [ ] Metrics are generated from real predictions.
-   [ ] Saved pipeline can be reloaded.
-   [ ] A sample DataFrame with the nine features can be predicted.

### API

-   [ ] `GET /api/health` works.
-   [ ] Valid prediction request returns HTTP 200.
-   [ ] Response keys and types match the documented contract.
-   [ ] Missing fields return HTTP 400.
-   [ ] Invalid categories return HTTP 400.
-   [ ] Out-of-range values return HTTP 400.
-   [ ] Decimal values are rejected for integer-only fields if not whole
    numbers.
-   [ ] The approval score corresponds to the `Approved` class.
-   [ ] Unexpected errors do not leak stack traces or filesystem paths.

### Frontend and integration

-   [ ] All nine fields appear in the correct order.
-   [ ] Exact API keys and category strings are used.
-   [ ] Submit sends one correctly formatted JSON request.
-   [ ] Loading state works.
-   [ ] Success state shows predicted class and approval score.
-   [ ] Error state is visible and understandable.
-   [ ] Failed requests do not leave stale results visible.
-   [ ] Reset works.
-   [ ] Frontend works through Flask using the relative API path.
-   [ ] The app works after a fresh start using the documented setup
    steps.

### Notebook/report

-   [ ] All notebook cells run top to bottom.
-   [ ] Figures are generated from the current cleaned dataset.
-   [ ] All four models have comparable metrics.
-   [ ] Holdout metrics are clearly distinguished from CV metrics.
-   [ ] Conclusions match the displayed results.
-   [ ] Synthetic-label limitation is stated.
-   [ ] No graph or metric is manually fabricated.

------------------------------------------------------------------------

## 14. Report and presentation sequence

Use the following order so the report, notebook, and presentation tell
the same story.

1.  Problem statement and objective
2.  Dataset source, size, target distribution, and synthetic-label
    limitation
3.  Data inspection and quality checks
4.  Feature selection and excluded columns
5.  Exploratory plots and observations
6.  Preprocessing and grouped train/test split
7.  Four selected algorithms and why they were compared
8.  Cross-validation and holdout evaluation method
9.  Comparison table and confusion matrices
10. Model-selection rationale
11. Flask architecture and frontend/API flow
12. Live demo
13. Limitations and future work
14. Conclusion

### Required figures

-   Target count plot
-   Numeric histograms
-   Categorical feature count plots
-   Claim amount by target boxplot
-   Target distribution by severity
-   Target distribution by police-report availability
-   Target distribution by authorities contacted
-   Numeric-feature correlation heatmap
-   Confusion matrix heatmap for each of four models
-   Holdout accuracy comparison
-   Holdout macro-F1 comparison
-   ROC curves/AUC comparison where supported and correctly calculated

Use readable fonts, consistent labels, clear legends, and meaningful
titles. Include a short interpretation for each important figure. Do not
include a graph merely for decoration.

### Viva explanation checklist

Every team member should be able to explain: - What the target is and
why it is synthetic - What `source_profile_id` does and why it is
excluded from model inputs - Why grouped splitting is required - What
one-hot encoding and scaling do - Why four models were compared - What
the confusion matrix shows - Why the final model was selected - What the
Flask API receives and returns - What the model can and cannot claim

------------------------------------------------------------------------

## 15. Definition of done

The project is ready for demonstration only when: 1. The notebook
follows the documented order and runs from top to bottom. 2. The grouped
split and evaluation are reproducible. 3. Four models have been compared
fairly. 4. The selected pipeline is saved and reloadable. 5. The Flask
API uses the exact nine-feature contract. 6. The frontend uses the same
exact categories and request/response keys. 7. All required smoke and
integration tests pass. 8. README setup instructions have been tested
from a clean environment. 9. Report figures and metrics match the actual
code outputs. 10. The team can explain the synthetic nature and
limitations of the target.

------------------------------------------------------------------------

## 16. Change-control rule

This document overrides informal chat messages and individual
assumptions.

If anyone believes a rule must change: 1. Open a pull request or raise
the proposed change with the project owner. 2. Explain why the change is
necessary. 3. Identify affected files, tests, notebook cells, API
fields, and report sections. 4. Update this README and all affected
tests in the same change. 5. Have the integrator approve the change
before merging.

Until approved, the current specification remains in force.

**The objective is a small, consistent, reproducible, explainable
college project---not the largest possible application.**
