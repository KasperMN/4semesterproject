import json
from flask import Flask
from flask_restful import Resource, Api, reqparse
import requests
from flask import request
import pandas as pd
import pandas.io
app = Flask(__name__)
api = Api(app)


@app.route('/data/<path:url>', methods=['GET'])
def getkeys(url):
    response = requests.get(url, verify=False).json()  # API call
    data = flatten_json(response)  # Flatten json, to get keys
    return data, 200


@app.route('/data', methods=['POST'])
def post():
    keys = request.json["keys"]  # Keys as list of column names
    keys
    return request.json, 200


def flatten_json(y):  # Magical flattening method
    out = {}
    print(type(y))

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
