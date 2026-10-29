"""
test_frontend_contract.py - Automated tests for Frontend HTML/Asset Contracts
Ensures templates/index.html, static/css/style.css, and static/js/script.js
strictly adhere to the specifications in README.md and PROGRESS.md.
"""

import os
import unittest
from html.parser import HTMLParser
from app import app, REQUIRED_FIELDS, ALLOWED_CATEGORIES, ALLOWED_DEDUCTIBLES


class FormHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inputs = {}
        self.selects = {}
        self.current_select = None
        self.has_disclaimer = False
        self.links = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)

        if tag == "link" and attr_dict.get("rel") == "stylesheet":
            self.links.append(attr_dict.get("href"))

        if tag == "script" and "src" in attr_dict:
            self.scripts.append(attr_dict.get("src"))

        if tag == "input":
            input_id = attr_dict.get("id")
            if input_id:
                self.inputs[input_id] = attr_dict

        if tag == "select":
            select_id = attr_dict.get("id")
            if select_id:
                self.current_select = select_id
                self.selects[select_id] = []

        if tag == "option" and self.current_select:
            val = attr_dict.get("value")
            if val is not None and val != "":
                self.selects[self.current_select].append(val)

    def handle_endtag(self, tag):
        if tag == "select":
            self.current_select = None

    def handle_data(self, data):
        if "synthetically generated" in data.lower() or "synthetic policy" in data.lower():
            self.has_disclaimer = True


class TestFrontendContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.root_dir = os.path.dirname(os.path.dirname(__file__))
        cls.template_path = os.path.join(cls.root_dir, "templates", "index.html")
        cls.css_path = os.path.join(cls.root_dir, "static", "css", "style.css")
        cls.js_path = os.path.join(cls.root_dir, "static", "js", "script.js")

        with open(cls.template_path, "r", encoding="utf-8") as f:
            cls.html_content = f.read()

        cls.parser = FormHTMLParser()
        cls.parser.feed(cls.html_content)

    def test_frontend_files_exist(self):
        self.assertTrue(os.path.exists(self.template_path), "templates/index.html does not exist")
        self.assertTrue(os.path.exists(self.css_path), "static/css/style.css does not exist")
        self.assertTrue(os.path.exists(self.js_path), "static/js/script.js does not exist")

    def test_flask_renders_index_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/html; charset=utf-8")
        self.assertIn("ClaimGuard AI", response.get_data(as_text=True))

    def test_all_nine_features_present_with_exact_ids(self):
        form_element_ids = set(self.parser.inputs.keys()) | set(self.parser.selects.keys())
        for field in REQUIRED_FIELDS:
            self.assertIn(field, form_element_ids, f"Required field ID '{field}' missing from form")

    def test_categorical_options_match_contract(self):
        for field, allowed_values in ALLOWED_CATEGORIES.items():
            self.assertIn(field, self.parser.selects, f"Select element for '{field}' not found")
            options = set(self.parser.selects[field])
            self.assertEqual(
                options,
                allowed_values,
                f"Dropdown options for '{field}' do not match expected categories",
            )

    def test_deductible_options_match_contract(self):
        self.assertIn("policy_deductable", self.parser.selects)
        options = {int(x) for x in self.parser.selects["policy_deductable"]}
        self.assertEqual(options, ALLOWED_DEDUCTIBLES)

    def test_integer_input_bounds_and_attributes(self):
        witnesses_attrs = self.parser.inputs.get("witnesses", {})
        self.assertEqual(witnesses_attrs.get("min"), "0")
        self.assertEqual(witnesses_attrs.get("max"), "3")
        self.assertEqual(witnesses_attrs.get("step"), "1")

        months_attrs = self.parser.inputs.get("months_as_customer", {})
        self.assertEqual(months_attrs.get("min"), "0")
        self.assertEqual(months_attrs.get("max"), "479")

        vehicles_attrs = self.parser.inputs.get("number_of_vehicles_involved", {})
        self.assertEqual(vehicles_attrs.get("min"), "1")
        self.assertEqual(vehicles_attrs.get("max"), "4")

        hour_attrs = self.parser.inputs.get("incident_hour_of_the_day", {})
        self.assertEqual(hour_attrs.get("min"), "0")
        self.assertEqual(hour_attrs.get("max"), "23")

    def test_disclaimer_present_in_template(self):
        self.assertTrue(self.parser.has_disclaimer, "Mandatory synthetic policy disclaimer missing from index.html")

    def test_static_assets_linked_and_served(self):
        # Verify index.html links to static assets
        self.assertIn("/static/css/style.css", self.parser.links)
        self.assertIn("/static/js/script.js", self.parser.scripts)

        # Verify Flask serves them correctly
        css_resp = self.client.get("/static/css/style.css")
        self.assertEqual(css_resp.status_code, 200)

        js_resp = self.client.get("/static/js/script.js")
        self.assertEqual(js_resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
