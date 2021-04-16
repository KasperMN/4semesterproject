import requests

from automated_classifier import app
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
    api_link = request.json["url"]
    chosen_columns = request.json["keys"]
    chosen_target = request.json["target"]
    chosen_table_name = request.json["table_name"]

    return app.find_best_classifier(
        link=api_link, columns=chosen_columns,
        target=chosen_target, table_name=chosen_table_name
    )

@_flask_application.route('data/prediction', methods=['POST'])
def predict_on_data():
    pass




