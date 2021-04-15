import unittest
from flask import json
from automated_classifier import api as ap
import warnings


class TestApi(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')

    def test_post_single_object(self):
        response = ap._flask_application.test_client().post(
            '/data', data=json.dumps(
                {
                    "keys": ["Area", "Volume", "Assembly Code"],
                    "url": "https://link.speckle.dk/api/objects/5f57e166f2ebf23b47b317af",
                    "target": "Assembly Code",
                    "table_name": "Walls"}), content_type='application/json')

        data = json.dumps(response.get_data(as_text=True))

        self.assertIsNotNone(data, "Data not none")
        self.assertEqual(response.status_code, 500, "Status Code 200")

    def test_post_all_objects(self):
        response = ap._flask_application.test_client().post(
            '/data', data=json.dumps(
                {
                    "keys": ["Area", "Volume", "Assembly Code"],
                    "url": "https://link.speckle.dk/api/streams/grB5WJmsE/objects",
                    "target": "Assembly Code",
                    "table_name": "Walls"}), content_type='application/json')

        data = json.dumps(response.get_data(as_text=True))

        self.assertIsNotNone(data, "Data not none")
        self.assertEqual(response.status_code, 200, "Status Code 200")


if __name__ == '__main__':
    unittest.main()
