import sqlite3

conn = sqlite3.connect("database.db")

cursor = conn.cursor()

cursor.execute("SELECT * FROM wals")

results = cursor.fetchall()
print(results)


class DBHandler:
    def __init__(self):
        self.conn = sqlite3.connect("database.db")
        self.cursor = self.conn.cursor()

    def create_tables(self):
        create1 = """CREATE TABLE IF NOT EXISTS
                    walls(Wall_ID char(30) PRIMARY KEY, Area float, 
                    Structutal bit, Volume float, Base_Constraint char(10), 
                    Assembly_Code int)
                    """
        cursor.execute(create1)

    def insert_rows(self):
        print(somethinjesperneeds.to_sql(name="wals", con=conn))