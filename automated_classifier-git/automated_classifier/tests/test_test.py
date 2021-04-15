from flask import json
from automated_classifier import api
import warnings
warnings.filterwarnings("ignore")


class TestApi:
    def test_post_single_object(self):
        response = api._flask_application.test_client().post(
            '/data', data=json.dumps(
                {
                    "keys": ["Area", "Volume", "Assembly Code"],
                    "url": "https://link.speckle.dk/api/objects/5f57e166f2ebf23b47b317af",
                    "target": "Assembly Code",
                    "table_name": "Walls"}), content_type='application/json')

        data = json.dumps(response.get_data(as_text=True))

        assert response.status_code == 200
        assert data is not None
