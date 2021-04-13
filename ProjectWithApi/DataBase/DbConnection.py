import sqlite3
from pandas import DataFrame


class DbConnection:
    def __init__(self, data: DataFrame):
        self.conn = sqlite3.connect(r"../DataBase/database.db")
        self.data = data
        self.data.to_sql(name="Table", con=self.conn)
