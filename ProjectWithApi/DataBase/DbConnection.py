import sqlite3
from pandas import DataFrame
import os
import pandas as pd


class DbConnection:
    def __init__(self, data: DataFrame, table_name: str):
        if os.path.exists(r"../DataBase/database.db"):
            os.remove(r"../DataBase/database.db")
        self.conn = sqlite3.connect(r"../DataBase/database.db")
        self.data = data
        self.table_name = table_name
        self.data.to_sql(name=self.table_name, con=self.conn)

    def collect_data(self, columns: list):
        sql_query = self.create_select_query_from_attribues(columns_to_select=columns)
        return pd.read_sql(sql=sql_query, con=self.conn)

    def create_select_query_from_attribues(self, columns_to_select: list):
        attribute_string = ''.join(
            [str("[" + column + '], ') if columns_to_select.index(column) != len(columns_to_select) - 1
            else str("[" + column + "]") for column in columns_to_select])

        return "Select {0} FROM [{1}]".format(attribute_string, self.table_name)