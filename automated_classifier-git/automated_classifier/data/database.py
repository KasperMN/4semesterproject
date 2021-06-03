import os
import sqlite3
from pandas import DataFrame
import pandas as pd


class Connection:
    def __init__(self, table_name: str, columns_to_select: list):
        self._conn = ''
        self._columns_to_select = columns_to_select
        self._table_name = table_name
        self._select_query = ''  # SELECT {column} FROM {table}

    @property
    def connection(self):
        return self._conn

    def create_select_query(self):
        select_string = ''.join(
            [str("[" + column + '], ') if self._columns_to_select.index(column) != len(self._columns_to_select) - 1
             else str("[" + column + "]") for column in self._columns_to_select])
        self._select_query = "Select {0} FROM [{1}]".format(select_string, self._table_name)

    def insert_data(self, data):
        data.to_sql(name=self._table_name, con=self._conn)

    def get_data(self):
        return pd.read_sql(sql=self._select_query, con=self._conn)

    def create_database(self):
        if os.path.exists(r"data/sql_lite.db"):
            os.remove(r"data/sql_lite.db")
        self._conn = sqlite3.connect(r"data/sql_lite.db")