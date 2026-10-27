"""
test_training_pipeline.py - Tests for Grouped Splitting, Training Pipeline, and Artifacts
"""

import json
import os
import unittest
import joblib
import numpy as np
import pandas as pd
from train_models import (
    MODEL_FEATURES,
    deterministic_grouped_split,
    load_and_validate_data,
    resolve_data_path,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


class TestTrainingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = resolve_data_path()
        cls.df = load_and_validate_data(data_path)

    def test_grouped_split_no_leakage(self):
        train_df, holdout_df, train_groups, holdout_groups = deterministic_grouped_split(self.df)
        self.assertEqual(len(train_groups), 792)
        self.assertEqual(len(holdout_groups), 197)
        self.assertEqual(len(train_df), 4005)
        self.assertEqual(len(holdout_df), 995)
        self.assertEqual(len(train_groups.intersection(holdout_groups)), 0)

    def test_features_exclude_source_profile_id(self):
        self.assertNotIn("source_profile_id", MODEL_FEATURES)
        self.assertNotIn("claim_status", MODEL_FEATURES)
        self.assertEqual(len(MODEL_FEATURES), 9)

    def test_model_artifact_exists_and_reloads(self):
        artifact_path = os.path.join(MODELS_DIR, "claim_model_pipeline.joblib")
        self.assertTrue(os.path.exists(artifact_path), "Serialized pipeline does not exist")

        pipeline = joblib.load(artifact_path)
        self.assertTrue(hasattr(pipeline, "predict"))
        self.assertTrue(hasattr(pipeline, "predict_proba"))
        self.assertEqual(set(pipeline.classes_), {"Approved", "Rejected"})

        sample_input = pd.DataFrame([{
            "witnesses": 2,
            "total_claim_amount": 45000,
            "policy_deductable": 1000,
            "months_as_customer": 120,
            "number_of_vehicles_involved": 2,
            "incident_hour_of_the_day": 14,
            "incident_severity": "Major Damage",
            "police_report_available": "YES",
            "authorities_contacted": "Police",
        }])

        pred = pipeline.predict(sample_input)[0]
        self.assertIn(pred, ["Approved", "Rejected"])
        probas = pipeline.predict_proba(sample_input)[0]
        self.assertEqual(len(probas), 2)
        self.assertTrue(0.0 <= probas[0] <= 1.0)
        self.assertTrue(0.0 <= probas[1] <= 1.0)

    def test_model_metadata_structure(self):
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        self.assertTrue(os.path.exists(meta_path), "model_metadata.json does not exist")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.assertIn("selected_model", meta)
        self.assertIn("features", meta)
        self.assertIn("split", meta)
        self.assertIn("performance", meta)

    def test_reports_comparison_json_exists(self):
        report_path = os.path.join(REPORTS_DIR, "model_comparison.json")
        self.assertTrue(os.path.exists(report_path), "model_comparison.json does not exist")
        with open(report_path, "r") as f:
            data = json.load(f)
        self.assertIn("models_evaluated", data)
        self.assertEqual(len(data["models_evaluated"]), 4)


if __name__ == "__main__":
    unittest.main()
