import sqlite3
from pandas import DataFrame
import os


class DbConnection:
    def __init__(self, data: DataFrame, table_name: str):
        if os.path.exists(r"../DataBase/database.db"):
            os.remove(r"../DataBase/database.db")
        self.conn = sqlite3.connect(r"../DataBase/database.db")
        self.data = data
        self.data.to_sql(name=table_name, con=self.conn)
