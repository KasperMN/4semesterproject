import requests

from automated_classifier.app import Application
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
    app = Application(request.json["url"], request.json["keys"], request.json["target"], request.json["table_name"])
    return app.return_best_model()

"""@_flask_application.route('data/prediction', methods=['POST'])
def predict_on_data():
    pass"""




