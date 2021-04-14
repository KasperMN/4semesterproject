from flask import Flask
from flask_restful import Resource, Api, reqparse
import requests
from DataCollector import DataCollector
from flask import request
import DataBase
import MachineLearning
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
api = Api(app)


def start_api():
    app.run()  # Initialize the api server


@app.route('/data/<path:url>', methods=['GET'])
def getkeys(url):
    response = requests.get(url, verify=False).json()  # API call
    data = flatten_json(response)  # Flatten json, to get keys
    return data, 200


@app.route('/data', methods=['POST'])
def post():
    api_link = request.json["url"]
    chosen_columns = request.json["keys"]
    target_to_classify = request.json["target"]
    database_table_name = request.json["table_name"]

    data_collector = DataCollector(keys=chosen_columns, url=api_link)
    error_messages, data = data_collector.collect_columns_from_keys(nested_dictionary=data_collector.nested_data)

    if not error_messages:  # If all went okay
        db = DataBase.DbConnection(data=data, table_name=database_table_name)  # Initialize Database with found data from keys
        data_from_db = db.collect_data(columns=chosen_columns)  # Fetches data from SqlLite database.db
        th = MachineLearning.TheHandler(data_from_db, target_to_classify)
        print(th.method())
        #prep = MachineLearning.PreProcessing(data=data_from_db)  # Preprocess the data
        #prep.returns_processed_test_and_training_data(target=target_to_classify)  # Processes the data
        #models = MachineLearning.Models(prep.get_data())

        return "DataBase Created, Preprocessing started...", 200  # Testing
    if error_messages:
        return error_messages, 400


def flatten_json(y):  # Magical flattening method
    out = {}

    def flatten(x, name=''):
        print(type(x))
        if type(x) is dict:
            for a in x:
                flatten(x[a], a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


if __name__ == '__main__':
    app.run()  # Initialize the api server
