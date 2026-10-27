"""
test_api.py - Comprehensive Tests for Flask Prediction API and Health Check

Follows Section 3.4 and Section 8 API Contract of README.md strictly:
- Exactly 9 required input fields
- Exact allowed categorical values
- Strict integer and float validation
- Rejection of extra/forbidden keys (source_profile_id, claim_status)
- Rejection of NaN, Inf, blanks, nulls, booleans for numbers
- Approval score calculation mapped directly to the pipeline's 'Approved' class index
- Safe 500 responses without leaking server internals or stack traces
"""

import json
import math
import unittest
from unittest.mock import patch
import pandas as pd

import app as app_module
from app import app, model_pipeline, model_name, REQUIRED_FIELDS


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.valid_payload = {
            "incident_severity": "Major Damage",
            "police_report_available": "YES",
            "authorities_contacted": "Police",
            "witnesses": 2,
            "total_claim_amount": 45000,
            "policy_deductable": 1000,
            "months_as_customer": 120,
            "number_of_vehicles_involved": 2,
            "incident_hour_of_the_day": 14,
        }

    # =========================================================================
    # 1. Health Endpoint Tests
    # =========================================================================
    def test_health_endpoint(self):
        """Health endpoint should return 200, status=healthy, and model metadata."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("model_loaded"))
        self.assertIsNotNone(data.get("model_name"))

    def test_health_endpoint_when_model_unloaded(self):
        """Health endpoint reports model_loaded=False when model pipeline is None."""
        with patch.object(app_module, "model_pipeline", None):
            response = self.client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data.get("status"), "healthy")
            self.assertFalse(data.get("model_loaded"))

    # =========================================================================
    # 2. Valid Prediction & Response Structure Tests
    # =========================================================================
    def test_valid_prediction(self):
        """Standard valid payload produces 200 OK with success contract."""
        response = self.client.post(
            "/api/predict",
            data=json.dumps(self.valid_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn(data.get("claim_status"), ["Approved", "Rejected"])
        self.assertIsInstance(data.get("approval_score"), (int, float))
        self.assertTrue(0.0 <= data.get("approval_score") <= 1.0)
        self.assertIsInstance(data.get("model_name"), str)

    def test_documented_response_schema_on_success(self):
        """Verify exact JSON response schema on successful prediction."""
        response = self.client.post(
            "/api/predict",
            data=json.dumps(self.valid_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        expected_keys = {"status", "claim_status", "approval_score", "model_name"}
        self.assertEqual(set(data.keys()), expected_keys)
        self.assertEqual(data["status"], "success")

    def test_documented_response_schema_on_validation_error(self):
        """Verify exact JSON response schema on validation error (HTTP 400)."""
        payload = self.valid_payload.copy()
        del payload["incident_severity"]
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        expected_keys = {"status", "error", "message"}
        self.assertEqual(set(data.keys()), expected_keys)
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"], "ValidationError")

    # =========================================================================
    # 3. Missing Fields Tests
    # =========================================================================
    def test_missing_field(self):
        """Payload missing incident_severity must be rejected with 400."""
        payload = self.valid_payload.copy()
        del payload["incident_severity"]
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertEqual(data.get("error"), "ValidationError")
        self.assertIn("Missing required fields", data.get("message"))

    def test_missing_multiple_fields(self):
        """Payload missing multiple fields must report all missing fields."""
        payload = {
            "incident_severity": "Major Damage",
            "witnesses": 1,
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("Missing required fields", data.get("message"))

    # =========================================================================
    # 4. Forbidden Extra Fields Tests (including source_profile_id & claim_status)
    # =========================================================================
    def test_forbidden_extra_field_source_profile_id(self):
        """source_profile_id is strictly a splitting key and forbidden in API."""
        payload = self.valid_payload.copy()
        payload["source_profile_id"] = "SP_123"
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertEqual(data.get("error"), "ValidationError")
        self.assertIn("Forbidden extra keys", data.get("message"))
        self.assertIn("source_profile_id", data.get("message"))

    def test_forbidden_extra_field_claim_status(self):
        """Target column claim_status must never be accepted in API input."""
        payload = self.valid_payload.copy()
        payload["claim_status"] = "Approved"
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertEqual(data.get("error"), "ValidationError")
        self.assertIn("Forbidden extra keys", data.get("message"))
        self.assertIn("claim_status", data.get("message"))

    # =========================================================================
    # 5. Null, Blank, or Incorrectly Typed Values
    # =========================================================================
    def test_null_values_rejected(self):
        """Null values in any field must be rejected with 400."""
        for field in self.valid_payload.keys():
            payload = self.valid_payload.copy()
            payload[field] = None
            response = self.client.post(
                "/api/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
            self.assertEqual(
                response.status_code,
                400,
                f"Field '{field}' with null value was not rejected with 400",
            )
            data = response.get_json()
            self.assertEqual(data.get("status"), "error")

    def test_blank_or_whitespace_strings_rejected(self):
        """Blank or whitespace-only strings must be rejected."""
        for field in ["incident_severity", "police_report_available", "authorities_contacted"]:
            for blank_val in ["", "   ", "\t\n"]:
                payload = self.valid_payload.copy()
                payload[field] = blank_val
                response = self.client.post(
                    "/api/predict",
                    data=json.dumps(payload),
                    content_type="application/json",
                )
                self.assertEqual(
                    response.status_code,
                    400,
                    f"Categorical field '{field}' with blank value '{blank_val}' was not rejected",
                )

    def test_boolean_for_numeric_fields_rejected(self):
        """Booleans (True/False) must not be coerced to 1/0 for integer or float fields."""
        numeric_fields = [
            "witnesses",
            "policy_deductable",
            "months_as_customer",
            "number_of_vehicles_involved",
            "incident_hour_of_the_day",
            "total_claim_amount",
        ]
        for field in numeric_fields:
            for bool_val in [True, False]:
                payload = self.valid_payload.copy()
                payload[field] = bool_val
                response = self.client.post(
                    "/api/predict",
                    data=json.dumps(payload),
                    content_type="application/json",
                )
                self.assertEqual(
                    response.status_code,
                    400,
                    f"Numeric field '{field}' allowed boolean {bool_val}",
                )

    def test_string_for_numeric_fields_rejected(self):
        """Strings supplied for numeric fields must be rejected."""
        payload = self.valid_payload.copy()
        payload["witnesses"] = "two"
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

        payload = self.valid_payload.copy()
        payload["total_claim_amount"] = "fifty-thousand"
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    # =========================================================================
    # 6. Unknown / Invalid Categorical Values
    # =========================================================================
    def test_invalid_category_incident_severity(self):
        payload = self.valid_payload.copy()
        payload["incident_severity"] = "Catastrophic"
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Invalid 'incident_severity'", data.get("message"))

    def test_invalid_category_police_report(self):
        payload = self.valid_payload.copy()
        payload["police_report_available"] = "Maybe"
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Invalid 'police_report_available'", data.get("message"))

    def test_invalid_category_authorities_contacted(self):
        payload = self.valid_payload.copy()
        payload["authorities_contacted"] = "Security"
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Invalid 'authorities_contacted'", data.get("message"))

    # =========================================================================
    # 7. NaN and Positive/Negative Infinity
    # =========================================================================
    def test_nan_total_claim_amount_rejected(self):
        """NaN total_claim_amount must be rejected."""
        payload = self.valid_payload.copy()
        payload["total_claim_amount"] = float("nan")
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("must be a finite number", data.get("message"))

    def test_infinity_total_claim_amount_rejected(self):
        """Positive and negative infinity must be rejected."""
        for inf_val in [float("inf"), float("-inf")]:
            payload = self.valid_payload.copy()
            payload["total_claim_amount"] = inf_val
            response = self.client.post(
                "/api/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertIn("must be a finite number", data.get("message"))

    def test_nan_and_infinity_in_integer_fields_rejected(self):
        """NaN and infinity in integer fields must be rejected."""
        for int_field in ["witnesses", "months_as_customer", "incident_hour_of_the_day"]:
            for special_val in [float("nan"), float("inf"), float("-inf")]:
                payload = self.valid_payload.copy()
                payload[int_field] = special_val
                response = self.client.post(
                    "/api/predict",
                    data=json.dumps(payload),
                    content_type="application/json",
                )
                self.assertEqual(
                    response.status_code,
                    400,
                    f"Field '{int_field}' accepted special float {special_val}",
                )

    # =========================================================================
    # 8. Fractional Values in Integer-Only Fields
    # =========================================================================
    def test_fractional_witnesses_rejected(self):
        payload = self.valid_payload.copy()
        payload["witnesses"] = 2.5
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("must be a whole integer", data.get("message"))

    def test_fractional_months_as_customer_rejected(self):
        payload = self.valid_payload.copy()
        payload["months_as_customer"] = 12.3
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("must be a whole integer", data.get("message"))

    def test_fractional_vehicles_involved_rejected(self):
        payload = self.valid_payload.copy()
        payload["number_of_vehicles_involved"] = 1.9
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("must be a whole integer", data.get("message"))

    def test_fractional_incident_hour_rejected(self):
        payload = self.valid_payload.copy()
        payload["incident_hour_of_the_day"] = 14.7
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("must be a whole integer", data.get("message"))

    def test_fractional_deductible_rejected(self):
        payload = self.valid_payload.copy()
        payload["policy_deductable"] = 500.5
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("must be a whole integer", data.get("message"))

    # =========================================================================
    # 9. Numeric Values Outside Permitted Bounds
    # =========================================================================
    def test_invalid_integer_bounds_witnesses(self):
        """witnesses allowed range [0, 3]."""
        for invalid_val in [-1, 4, 10]:
            payload = self.valid_payload.copy()
            payload["witnesses"] = invalid_val
            response = self.client.post(
                "/api/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertIn("outside allowed range", data.get("message"))

    def test_invalid_integer_bounds_months_as_customer(self):
        """months_as_customer allowed range [0, 479]."""
        for invalid_val in [-1, 480, 1000]:
            payload = self.valid_payload.copy()
            payload["months_as_customer"] = invalid_val
            response = self.client.post(
                "/api/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 400)

    def test_invalid_integer_bounds_number_of_vehicles(self):
        """number_of_vehicles_involved allowed range [1, 4]."""
        for invalid_val in [0, 5, -2]:
            payload = self.valid_payload.copy()
            payload["number_of_vehicles_involved"] = invalid_val
            response = self.client.post(
                "/api/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 400)

    def test_invalid_integer_bounds_incident_hour(self):
        """incident_hour_of_the_day allowed range [0, 23]."""
        for invalid_val in [-1, 24, 30]:
            payload = self.valid_payload.copy()
            payload["incident_hour_of_the_day"] = invalid_val
            response = self.client.post(
                "/api/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 400)

    def test_total_claim_amount_below_and_above_bounds(self):
        """total_claim_amount supported range [$98, $116602]."""
        for invalid_amt in [-100.0, 0.0, 50.0, 97.99, 116602.01, 200000.0]:
            payload = self.valid_payload.copy()
            payload["total_claim_amount"] = invalid_amt
            response = self.client.post(
                "/api/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
            self.assertEqual(
                response.status_code,
                400,
                f"total_claim_amount ${invalid_amt} was not rejected",
            )
            data = response.get_json()
            self.assertIn("outside the supported range", data.get("message"))

    # =========================================================================
    # 10. Invalid Policy Deductible Values
    # =========================================================================
    def test_invalid_deductible(self):
        """policy_deductable only allows 500, 1000, 2000."""
        for invalid_deductible in [0, 250, 750, 1500, 3000, -500]:
            payload = self.valid_payload.copy()
            payload["policy_deductable"] = invalid_deductible
            response = self.client.post(
                "/api/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
            self.assertEqual(
                response.status_code,
                400,
                f"Deductible {invalid_deductible} was not rejected",
            )
            data = response.get_json()
            self.assertIn("Invalid 'policy_deductable'", data.get("message"))

    # =========================================================================
    # 11. Valid Boundary Values (Min and Max)
    # =========================================================================
    def test_valid_min_boundary_values(self):
        """All lowest allowed values should succeed (200 OK)."""
        payload = {
            "incident_severity": "Trivial Damage",
            "police_report_available": "Unknown",
            "authorities_contacted": "None Contacted",
            "witnesses": 0,
            "total_claim_amount": 98.0,
            "policy_deductable": 500,
            "months_as_customer": 0,
            "number_of_vehicles_involved": 1,
            "incident_hour_of_the_day": 0,
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn(data.get("claim_status"), ["Approved", "Rejected"])

    def test_valid_max_boundary_values(self):
        """All highest allowed values should succeed (200 OK)."""
        payload = {
            "incident_severity": "Total Loss",
            "police_report_available": "NO",
            "authorities_contacted": "Ambulance",
            "witnesses": 3,
            "total_claim_amount": 116602.0,
            "policy_deductable": 2000,
            "months_as_customer": 479,
            "number_of_vehicles_involved": 4,
            "incident_hour_of_the_day": 23,
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn(data.get("claim_status"), ["Approved", "Rejected"])

    # =========================================================================
    # 12. Approval Score Mapping from Saved Pipeline
    # =========================================================================
    def test_approval_score_matches_pipeline_approved_class(self):
        """
        Verify approval_score corresponds strictly to the 'Approved' class
        probability from pipeline.classes_, not an inverted or arbitrary index.
        """
        response = self.client.post(
            "/api/predict",
            data=json.dumps(self.valid_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        api_score = data.get("approval_score")

        # Manually compute probability using loaded pipeline
        input_df = pd.DataFrame([self.valid_payload])[REQUIRED_FIELDS]
        classes = list(model_pipeline.classes_)
        approved_idx = classes.index("Approved")
        expected_score = float(model_pipeline.predict_proba(input_df)[0][approved_idx])

        self.assertAlmostEqual(api_score, expected_score, places=5)
        self.assertTrue(0.0 <= api_score <= 1.0)

    # =========================================================================
    # 13. Unexpected Errors & Safe 500 Responses
    # =========================================================================
    def test_unexpected_error_returns_safe_500_without_leak(self):
        """
        When model prediction raises an unhandled exception, API must return
        HTTP 500 with a generic message and NO stack traces, sensitive paths, or debug logs.
        """
        secret_leak = "C:\\Sensitive\\Internal\\Database\\secret_key.pem"
        with patch.object(
            model_pipeline,
            "predict",
            side_effect=RuntimeError(f"Unexpected database failure: {secret_leak}"),
        ):
            response = self.client.post(
                "/api/predict",
                data=json.dumps(self.valid_payload),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 500)
            data = response.get_json()
            self.assertEqual(data.get("status"), "error")
            self.assertEqual(data.get("error"), "InternalServerError")
            self.assertEqual(
                data.get("message"),
                "An unexpected error occurred while processing the prediction.",
            )

            # Assert complete text does not leak internal path or traceback
            raw_text = response.get_data(as_text=True)
            self.assertNotIn("Traceback", raw_text)
            self.assertNotIn("RuntimeError", raw_text)
            self.assertNotIn(secret_leak, raw_text)
            self.assertNotIn("Sensitive", raw_text)

    def test_unloaded_model_returns_500_server_error(self):
        """If model_pipeline is None and cannot be loaded, API returns 500 ServerError."""
        with patch.object(app_module, "model_pipeline", None), patch.object(
            app_module, "load_model"
        ):
            response = self.client.post(
                "/api/predict",
                data=json.dumps(self.valid_payload),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 500)
            data = response.get_json()
            self.assertEqual(data.get("status"), "error")
            self.assertEqual(data.get("error"), "ServerError")
            self.assertIn("Model pipeline is not loaded", data.get("message"))

    # =========================================================================
    # 14. Non-JSON and Bad Request Bodies
    # =========================================================================
    def test_non_json_request(self):
        """Non-JSON request body should return 400 ValidationError."""
        response = self.client.post(
            "/api/predict",
            data="not json",
            content_type="text/plain",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertEqual(data.get("error"), "ValidationError")

    def test_malformed_json_syntax(self):
        """Malformed JSON payload string should return 400 ValidationError."""
        response = self.client.post(
            "/api/predict",
            data='{"incident_severity": "Major Damage", broken_json',
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertEqual(data.get("error"), "ValidationError")

    def test_non_dictionary_json_payload(self):
        """JSON payload that is a list or primitive instead of object returns 400."""
        response = self.client.post(
            "/api/predict",
            data=json.dumps(["not", "a", "dict"]),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("must be a valid JSON object", data.get("message"))


if __name__ == "__main__":
    unittest.main()
