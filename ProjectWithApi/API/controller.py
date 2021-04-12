import json
from flask import Flask
from flask_restful import Resource, Api, reqparse
import requests
import pandas as pd
import pandas.io
app = Flask(__name__)
api = Api(app)


@app.route('/data/<path:url>', methods=['GET'])
def get(url):
    response = requests.get(url, verify=False).json()
    data = flatten_json(response)



    """keys = set()
    for i, element in enumerate(df):
        print(i, element)
        for i, element in enumerate(df[element]):
            if isinstance(element, dict):
                print(element)"""

    #print(df.keys())

    return data, 200


def flatten_json(y):
    out = {}

    def flatten(x, name=''):
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
