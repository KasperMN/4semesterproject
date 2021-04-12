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

    def collect_columns_from_keys(self, hej):
        print(type(hej))
        for i, element_i in enumerate(hej):
            for j, element_j in enumerate(element_i):
                if type(element_j) is dict:
                    for a in element_j:
                        print("IF")
                        if a in self.keys:
                            self.values[a].append(element_j[a])
                        self.collect_columns_from_keys(element_j[a])
                elif type(element_j) is list:
                    for a in element_j:
                        print("ELSE IF")
                        if a in self.keys:
                            self.values[a].append(element_j[a])
                        self.collect_columns_from_keys(element_j[a])

        print(self.values)

    """def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + '_')
                i += 1
        else:
            out[name[:-1]] = x"""


if __name__ == '__main__':
    dc = DataCollector(keys=["Area", "Volume", "Length"], url="hej")
    url = "https://link.speckle.dk/api/streams/grB5WJmsE/objects"
    JSONContent = requests.get(url, verify=False).json()
    content = json.dumps(JSONContent, indent=4, sort_keys=True)
    df = pd.read_json(content)
    dc.collect_columns_from_keys(df)
