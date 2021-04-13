import json

import pandas as pd
import requests
from pandas import DataFrame


class DataCollector:
    def __init__(self, keys: list, url):
        self.keys = keys
        self.url = url

        self.values = {}
        for key in keys:
            self.values[key] = []

        """self.url = "https://link.speckle.dk/api/streams/grB5WJmsE/objects"
        self.JSONContent = requests.get(self.url, verify=False).json()
        self.content = json.dumps(self.JSONContent, indent=4, sort_keys=True)
        self.df = pd.read_json(self.content)"""

    def collect_columns_from_keys(self, dic):
        df = pd.DataFrame()
        for x in self.values:
            df[x] = json_extract(dic, x)

        print(df)


# Magical json extract method
def json_extract(obj, key):
    """Recursively fetch values from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values


if __name__ == '__main__':
    dc = DataCollector(keys=["Area", "Volume", "Length"], url="hej")
    with open(r"../../Data/data2.json") as json_file:
        url = json.load(json_file)
    dc.collect_columns_from_keys(url)
