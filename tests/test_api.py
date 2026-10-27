"""
test_api.py - Tests for Flask Prediction API and Health Check
"""

import json
import unittest
from app import app


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

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("model_loaded"))
        self.assertIsNotNone(data.get("model_name"))

    def test_valid_prediction(self):
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

    def test_missing_field(self):
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

    def test_forbidden_extra_field(self):
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

    def test_invalid_category(self):
        payload = self.valid_payload.copy()
        payload["incident_severity"] = "Catastrophic"
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertEqual(data.get("error"), "ValidationError")
        self.assertIn("Invalid 'incident_severity'", data.get("message"))

    def test_invalid_integer_bound(self):
        payload = self.valid_payload.copy()
        payload["witnesses"] = 5  # Allowed is 0-3
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("outside allowed range", data.get("message"))

    def test_decimal_in_integer_field(self):
        payload = self.valid_payload.copy()
        payload["witnesses"] = 2.5
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("must be a whole integer", data.get("message"))

    def test_invalid_deductible(self):
        payload = self.valid_payload.copy()
        payload["policy_deductable"] = 750  # Only 500, 1000, 2000 allowed
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("Invalid 'policy_deductable'", data.get("message"))

    def test_non_json_request(self):
        response = self.client.post(
            "/api/predict",
            data="not json",
            content_type="text/plain",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")


if __name__ == "__main__":
    unittest.main()
