import requests
from automated_classifier import api
import unittest


class TestApi(unittest.TestCase):
    def test_post(self):
        api.run_tests()

        j_son_data = {
            "keys": ["Area", "Volume", "Assembly Code"],
            "url": "https://link.speckle.dk/api/objects/5f57e166f2ebf23b47b317af",
            "target": "Assembly Code",
            "table_name": "Walls"
        }

        response = requests.get('http://127.0.0.1:5000/data', data=j_son_data)
        print(response.json())
        self.assertIsNotNone(response.json())

    def test_something(self):

        self.assertEqual(2, 2)
