import pandas as pd
import requests

from automated_classifier.app import TrainerApp
from automated_classifier.app import PredictApp
from flask import Flask
from flask_restful import Api, Resource
from flask import request
from automated_classifier.common import extractors
import warnings

warnings.filterwarnings("ignore")

_flask_application = Flask(__name__)
api = Api(_flask_application)


def run_restful_api():
    _flask_application.run()


@_flask_application.route('/data/<path:url>', methods=['GET'])
def get_keys(url):
    try:
        response = requests.get(url, verify=False).json()  # API call
        data = extractors.flatten_json(response)
        keys_list = []
        for key in data.keys():
            if key not in keys_list:
                keys_list.append(key)


    except Exception:
        return Exception, 400
    return {"Column Names": keys_list}


@_flask_application.route('/data', methods=['POST'])
def returns_model():
    app = TrainerApp(request.json["url"], request.json["keys"], request.json["target"], request.json["table_name"])
    return app.return_best_model()


@_flask_application.route('/predict', methods=['POST'])
def predict_on_data():
    print(request.json["params"])
    df = pd.DataFrame(request.json["params"], index=[0])

    app = PredictApp()
    return app.predict_classification(df), 200


@_flask_application.route('/test/predict/dataset', methods=['POST'])
def predict_on_dataset():
    app = PredictApp()
    app.test_predict_dataset()
    return "ok", 200
