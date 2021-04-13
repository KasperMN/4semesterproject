from flask import Flask
from flask_restful import Resource, Api, reqparse
import requests
from DataCollector import DataCollector
from flask import request
app = Flask(__name__)
api = Api(app)


def start_api():
    app.run()  # Initialize the api server

@app.route('/data/<path:url>', methods=['GET'])
def getkeys(url):
    response = requests.get(url, verify=False).json()  # API call
    data = flatten_json(response)  # Flatten json, to get keys
    return data, 200


@app.route('/data/<path:url>', methods=['POST'])
def post(url):
    dt = DataCollector(keys=request.json, url=url)
    test = dt.collect_columns_from_keys(dic=dt.response)
    return test, 200


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
