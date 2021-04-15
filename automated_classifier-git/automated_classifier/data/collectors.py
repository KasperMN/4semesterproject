from collections import defaultdict
from automated_classifier.common import json_extractor
import pandas
import requests


class DataCollector:
    def __init__(self, chosen_columns: list):
        self._chosen_columns = chosen_columns  # Chosen columns to find in dataset
        self._api_data = defaultdict()  # Api Call - nested_data as dict with json
        self._chosen_data = pandas.DataFrame()  # Chosen data from api
        self._error_messages = defaultdict()  # Attribute error messages

    @property
    def api_data(self):
        return self._api_data

    @api_data.setter
    def api_data(self, api_link):
        self._api_data = requests.get(api_link, verify=False).json()

    @property
    def chosen_data(self) -> pandas.DataFrame:
        for column in self._chosen_columns:
            try:
                self._chosen_data[column] = json_extractor(self._api_data, column)
                if self._chosen_data.shape[0] < 100:
                    self._error_messages["Attribute error: {}".format(column)] = "Too few rows"
            except ValueError:
                if self._chosen_data not in self._error_messages:
                    self._error_messages["Attribute error: {}".format(column)] \
                        = "Too many rows compared to other attributes"

        return self._chosen_data

    @property
    def error_messages(self):
        return self._error_messages
