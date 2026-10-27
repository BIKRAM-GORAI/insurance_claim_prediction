"""
test_data_contract.py - Tests for Dataset Contract and Integrity
"""

import os
import unittest
import pandas as pd
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "insurance_claim_dataset_cleaned.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "insurance_claim_dataset_cleaned.csv")

EXPECTED_FEATURES = [
    "witnesses",
    "total_claim_amount",
    "policy_deductable",
    "months_as_customer",
    "number_of_vehicles_involved",
    "incident_hour_of_the_day",
    "incident_severity",
    "police_report_available",
    "authorities_contacted",
]


class TestDataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(CSV_PATH)

    def test_file_exists_and_shape(self):
        self.assertTrue(os.path.exists(CSV_PATH), "Dataset file does not exist")
        self.assertEqual(self.df.shape, (5000, 40), f"Unexpected dataset shape: {self.df.shape}")

    def test_target_column_and_labels(self):
        self.assertIn("claim_status", self.df.columns)
        targets = set(self.df["claim_status"].dropna().unique())
        self.assertEqual(targets, {"Approved", "Rejected"})
        counts = self.df["claim_status"].value_counts().to_dict()
        self.assertEqual(counts["Approved"], 3042)
        self.assertEqual(counts["Rejected"], 1958)

    def test_source_profile_id_exists(self):
        self.assertIn("source_profile_id", self.df.columns)
        self.assertEqual(self.df["source_profile_id"].nunique(), 989)

    def test_nine_features_present_and_no_nulls(self):
        for feature in EXPECTED_FEATURES:
            self.assertIn(feature, self.df.columns, f"Missing feature: {feature}")
        null_counts = self.df[EXPECTED_FEATURES + ["claim_status", "source_profile_id"]].isnull().sum()
        self.assertEqual(null_counts.sum(), 0, "Null values detected in core features")

    def test_categorical_allowed_values(self):
        allowed_severity = {"Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"}
        allowed_police = {"YES", "NO", "Unknown"}
        allowed_auth = {"Police", "Fire", "Ambulance", "Other", "None Contacted"}

        self.assertTrue(set(self.df["incident_severity"].unique()).issubset(allowed_severity))
        self.assertTrue(set(self.df["police_report_available"].unique()).issubset(allowed_police))
        self.assertTrue(set(self.df["authorities_contacted"].unique()).issubset(allowed_auth))

    def test_numeric_ranges(self):
        self.assertTrue((self.df["witnesses"] >= 0).all() and (self.df["witnesses"] <= 3).all())
        self.assertTrue((self.df["months_as_customer"] >= 0).all() and (self.df["months_as_customer"] <= 479).all())
        self.assertTrue((self.df["number_of_vehicles_involved"] >= 1).all() and (self.df["number_of_vehicles_involved"] <= 4).all())
        self.assertTrue((self.df["incident_hour_of_the_day"] >= 0).all() and (self.df["incident_hour_of_the_day"] <= 23).all())
        self.assertEqual(set(self.df["policy_deductable"].unique()), {500, 1000, 2000})

    def test_claim_amount_equals_sum_of_components(self):
        diff = (self.df["total_claim_amount"] - (
            self.df["injury_claim"] + self.df["property_claim"] + self.df["vehicle_claim"]
        )).abs()
        self.assertEqual(diff.max(), 0, "total_claim_amount does not match sum of claim components")


if __name__ == "__main__":
    unittest.main()
