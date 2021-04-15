from automated_classifier import app
from flask import Flask
from flask_restful import Api, Resource
from flask import request
import warnings
warnings.filterwarnings("ignore")

_flask_application = Flask(__name__)
api = Api(_flask_application)


def run_restful_api():
    _flask_application.run()


def run_tests():
    _flask_application.run(host='localhost', debug=True, use_reloader=False)


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





