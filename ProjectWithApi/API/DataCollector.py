import json
import sys

import pandas as pd
import requests


class DataCollector:
    def __init__(self, keys: dict, url):
        self.keys = keys
        self.response = requests.get(url, verify=False).json()

    def collect_columns_from_keys(self, nested_dictionary):
        df = pd.DataFrame()
        error_messages = {}
        for x in self.keys:
            try:
                df[x] = json_extract(nested_dictionary, x)

            except:
                if x not in error_messages:
                    error_messages[x] = "This Attribute Is Not Valid - Chose another or remove it"

        return error_messages, df


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
