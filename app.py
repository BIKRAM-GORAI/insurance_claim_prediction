"""
app.py - Flask Web Application and Prediction API

Implements:
- GET /api/health : Health check confirming process and model load status
- POST /api/predict : Validates 9 inputs against data contract and returns prediction
- GET / : Serves frontend index.html template
"""

import json
import math
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Model artifact path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "claim_model_pipeline.joblib")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "models", "model_metadata.json")

# Permitted features and strict domain constraints
ALLOWED_CATEGORIES = {
    "incident_severity": {"Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"},
    "police_report_available": {"YES", "NO", "Unknown"},
    "authorities_contacted": {"Police", "Fire", "Ambulance", "Other", "None Contacted"},
}

INTEGER_BOUNDS = {
    "witnesses": (0, 3),
    "months_as_customer": (0, 479),
    "number_of_vehicles_involved": (1, 4),
    "incident_hour_of_the_day": (0, 23),
}

ALLOWED_DEDUCTIBLES = {500, 1000, 2000}
TOTAL_CLAIM_AMOUNT_BOUNDS = (98.0, 116602.0)

REQUIRED_FIELDS = [
    "incident_severity",
    "police_report_available",
    "authorities_contacted",
    "witnesses",
    "total_claim_amount",
    "policy_deductable",
    "months_as_customer",
    "number_of_vehicles_involved",
    "incident_hour_of_the_day",
]

# Global cache for loaded model and metadata
model_pipeline = None
model_name = "Unknown"


def load_model():
    """Load the serialized scikit-learn pipeline and metadata."""
    global model_pipeline, model_name
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "r") as f:
                    meta = json.load(f)
                    model_name = meta.get("selected_model", "Unknown")
            else:
                model_name = model_pipeline.named_steps.get("classifier").__class__.__name__
            print(f"Model successfully loaded: {model_name}")
        except Exception as e:
            print(f"Error loading model pipeline: {e}")
            model_pipeline = None


# Attempt model load on import/startup
load_model()


def validate_prediction_payload(data):
    """
    Validate incoming JSON payload strictly against Section 3.4 & 8.5 of README.
    Returns (cleaned_dict, error_message) tuple.
    """
    if not isinstance(data, dict):
        return None, "Request body must be a valid JSON object."

    # Reject unexpected keys (e.g. source_profile_id, claim_status)
    payload_keys = set(data.keys())
    expected_keys = set(REQUIRED_FIELDS)
    extra_keys = payload_keys - expected_keys
    if extra_keys:
        return None, f"Forbidden extra keys in payload: {sorted(list(extra_keys))}."

    missing_keys = expected_keys - payload_keys
    if missing_keys:
        return None, f"Missing required fields: {sorted(list(missing_keys))}."

    cleaned = {}

    # Validate categorical features
    for cat_field, allowed_values in ALLOWED_CATEGORIES.items():
        val = data.get(cat_field)
        if val is None or not isinstance(val, str) or val.strip() == "":
            return None, f"Field '{cat_field}' must be a non-empty string."
        if val not in allowed_values:
            return (
                None,
                f"Invalid '{cat_field}'. Allowed values: {', '.join(sorted(allowed_values))}.",
            )
        cleaned[cat_field] = val

    # Validate integer fields
    for int_field, (min_v, max_v) in INTEGER_BOUNDS.items():
        val = data.get(int_field)
        if val is None or isinstance(val, bool):
            return None, f"Field '{int_field}' must be an integer between {min_v} and {max_v}."
        if isinstance(val, float):
            if not val.is_integer():
                return (
                    None,
                    f"Field '{int_field}' must be a whole integer, got decimal {val}.",
                )
            val = int(val)
        elif not isinstance(val, int):
            return None, f"Field '{int_field}' must be an integer between {min_v} and {max_v}."

        if not (min_v <= val <= max_v):
            return (
                None,
                f"Field '{int_field}' value {val} is outside allowed range [{min_v}, {max_v}].",
            )
        cleaned[int_field] = val

    # Validate policy_deductable
    deductible = data.get("policy_deductable")
    if deductible is None or isinstance(deductible, bool):
        return None, "Field 'policy_deductable' must be an integer (500, 1000, or 2000)."
    if isinstance(deductible, float):
        if not deductible.is_integer():
            return None, "Field 'policy_deductable' must be a whole integer (500, 1000, or 2000)."
        deductible = int(deductible)
    elif not isinstance(deductible, int):
        return None, "Field 'policy_deductable' must be an integer (500, 1000, or 2000)."

    if deductible not in ALLOWED_DEDUCTIBLES:
        return (
            None,
            f"Invalid 'policy_deductable'. Allowed values: {sorted(list(ALLOWED_DEDUCTIBLES))}.",
        )
    cleaned["policy_deductable"] = deductible

    # Validate total_claim_amount
    claim_amount = data.get("total_claim_amount")
    if claim_amount is None or isinstance(claim_amount, bool):
        return None, "Field 'total_claim_amount' must be a positive numeric amount."
    if not isinstance(claim_amount, (int, float)):
        return None, "Field 'total_claim_amount' must be numeric."

    claim_amount = float(claim_amount)
    if math.isnan(claim_amount) or math.isinf(claim_amount):
        return None, "Field 'total_claim_amount' must be a finite number."

    min_amount, max_amount = TOTAL_CLAIM_AMOUNT_BOUNDS
    if not (min_amount <= claim_amount <= max_amount):
        return (
            None,
            f"Field 'total_claim_amount' ${claim_amount:.2f} is outside the supported range [${min_amount:.2f}, ${max_amount:.2f}].",
        )
    cleaned["total_claim_amount"] = claim_amount

    return cleaned, None


@app.route("/")
def index():
    """Serve frontend index.html template or API status info."""
    template_path = os.path.join(app.root_path, "templates", "index.html")
    if os.path.exists(template_path):
        return render_template("index.html")
    return jsonify({
        "message": "Insurance Claim Prediction API is running. Presentation layer (index.html) will be provided by frontend PR.",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict"
        }
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "model_name": model_name,
    }), 200


def compute_feature_influences(pipeline, input_df):
    """
    Computes genuine mathematical feature contributions for the prediction.
    For linear models (Logistic Regression), decomposes log-odds across the 9 features.
    """
    try:
        clf = pipeline.named_steps.get("classifier")
        prep = pipeline.named_steps.get("preprocessor")
        if clf is None or prep is None or not hasattr(clf, "coef_"):
            return []

        transformed = prep.transform(input_df)[0]
        feature_names = prep.get_feature_names_out()

        # In sklearn binary classification:
        # Decision function is for classes_[1].
        # For class 'Approved', direction is inverted if Approved is classes_[0].
        classes = list(pipeline.classes_)
        approved_idx = classes.index("Approved")
        raw_coef = clf.coef_[0] if approved_idx == 1 else -clf.coef_[0]
        contributions = transformed * raw_coef

        field_contributions = {}
        for fname, val in zip(feature_names, contributions):
            if fname.startswith("num__"):
                col = fname.replace("num__", "")
                field_contributions[col] = field_contributions.get(col, 0.0) + float(val)
            elif fname.startswith("cat__"):
                for col in ["incident_severity", "police_report_available", "authorities_contacted"]:
                    if col in fname:
                        field_contributions[col] = field_contributions.get(col, 0.0) + float(val)
                        break

        human_labels = {
            "police_report_available": lambda v: f"Police Report: {v}",
            "incident_severity": lambda v: f"Damage Severity: {v}",
            "authorities_contacted": lambda v: f"Authorities: {v}",
            "total_claim_amount": lambda v: f"Claim Amount: ${v:,.2f}",
            "witnesses": lambda v: f"Witnesses: {v}",
            "months_as_customer": lambda v: f"Tenure: {v} months",
            "policy_deductable": lambda v: f"Deductible: ${v}",
            "number_of_vehicles_involved": lambda v: f"Vehicles: {v}",
            "incident_hour_of_the_day": lambda v: f"Time: {v:02d}:00",
        }

        row = input_df.iloc[0]
        results = []
        for field, score in sorted(field_contributions.items(), key=lambda x: abs(x[1]), reverse=True):
            raw_val = row.get(field)
            val = raw_val.item() if hasattr(raw_val, "item") else raw_val
            label = human_labels[field](val) if field in human_labels else f"{field}: {val}"

            if field == "police_report_available":
                explanation = "Official police report on file" if val == "YES" else ("No police report available" if val == "NO" else "Report status unconfirmed")
            elif field == "incident_severity":
                explanation = "Severe vehicle damage scale" if val in ["Major Damage", "Total Loss"] else "Minor / non-critical damage scale"
            elif field == "authorities_contacted":
                explanation = "Emergency response unit contacted" if val in ["Police", "Fire", "Ambulance"] else ("No authorities contacted" if val == "None Contacted" else "Other responders involved")
            elif field == "total_claim_amount":
                explanation = "High reimbursement payout requested" if score < 0 else "Moderate claim amount within profile bounds"
            elif field == "witnesses":
                explanation = f"{val} corroborating witness(es) on scene" if val > 0 else "Zero independent witnesses"
            elif field == "months_as_customer":
                explanation = f"Account history of {val} months"
            else:
                explanation = "Supports approval" if score > 0 else "Increases rejection risk"

            results.append({
                "field": field,
                "label": label,
                "value": val,
                "contribution": round(score, 2),
                "direction": "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral"),
                "explanation": explanation,
            })
        return results
    except Exception as e:
        print(f"Error computing feature influences: {e}")
        return []


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts 9 input features, returns model-predicted claim_status, approval_score,
    and genuine mathematical feature contributions.
    """
    global model_pipeline
    if model_pipeline is None:
        load_model()
        if model_pipeline is None:
            return jsonify({
                "status": "error",
                "error": "ServerError",
                "message": "Model pipeline is not loaded on server. Please train the model first.",
            }), 500

    if not request.is_json:
        return jsonify({
            "status": "error",
            "error": "ValidationError",
            "message": "Request Content-Type must be application/json.",
        }), 400

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "status": "error",
            "error": "ValidationError",
            "message": "Invalid JSON format in request body.",
        }), 400

    cleaned_data, error_msg = validate_prediction_payload(payload)
    if error_msg:
        return jsonify({
            "status": "error",
            "error": "ValidationError",
            "message": error_msg,
        }), 400

    try:
        # Prepare DataFrame with required column ordering
        input_df = pd.DataFrame([cleaned_data])[REQUIRED_FIELDS]

        # Predict label
        predicted_class = model_pipeline.predict(input_df)[0]

        # Dynamic score for 'Approved' class from pipeline.classes_
        classes = list(model_pipeline.classes_)
        approved_idx = classes.index("Approved")
        probabilities = model_pipeline.predict_proba(input_df)[0]
        approval_score = float(probabilities[approved_idx])

        # Compute genuine mathematical feature influences
        influences = compute_feature_influences(model_pipeline, input_df)

        return jsonify({
            "status": "success",
            "claim_status": str(predicted_class),
            "approval_score": approval_score,
            "model_name": model_name,
            "influencing_factors": influences,
        }), 200

    except Exception as e:
        # Catch unexpected errors without leaking filesystem details or stack traces
        return jsonify({
            "status": "error",
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing the prediction.",
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
