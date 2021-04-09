import sqlite3
import pandas as pd
import database.connection


class DBHandler:
    def __init__(self):
        self.conn = sqlite3.connect("database.db")
        self.cursor = self.conn.cursor()
        self.create_extra_tables()
        self.api = database.connection.ApiConnection()
        self.data = self.api.collect_link_data()
        self.data.to_sql(name="walls", con=self.conn)
        self.add_foreign_key()

        self.cursor.execute("SELECT * FROM walls")
        print(self.cursor.fetchall())

    def create_extra_tables(self):
        create_company = """CREATE TABLE IF NOT EXISTS
        companies(company_id INTEGER PRIMARY KEY AUTOINCREMENT, name CHAR(100))"""
        create_projects = """CREATE TABLE IF NOT EXISTS
        projects(project_id INTEGER PRIMARY KEY AUTOINCREMENT, name CHAR(100), 
        company_id INTEGER, FOREIGN KEY (company_id) REFERENCES companies(company_id))"""

        insert_companies = """INSERT INTO companies VALUES(1, 'LINK Arkitektur')"""
        insert_projects = """INSERT INTO projects VALUES(1, 'Skejby Hospital', 1)"""

        self.cursor.execute(create_company)
        self.cursor.execute(insert_companies)
        self.cursor.execute(create_projects)

        self.cursor.execute(insert_projects)

    def add_foreign_key(self):
        sql = """PRAGMA foreign_keys=off;
        
        BEGIN TRANSACTION;
        
        ALTER TABLE walls RENAME TO old_walls;
        
        CREATE TABLE walls (
           [index] CHAR(30) NOT NULL PRIMARY KEY,
           [Assembly_Code] INTEGER NOT NULL,
           [Area] REAL,
           [Structural] BOOLEAN,
           [Volume] REAL,
           [Base_Constraint] CHAR(30),
           [Project_id] INTEGER,
           FOREIGN KEY (Project_id) REFERENCES projects(project_id)
        );
        
        INSERT INTO walls 
        SELECT * FROM old_walls;
        
        DROP TABLE old_walls;
        
        COMMIT;
        
        PRAGMA foreign_keys=on;"""
        self.cursor.executescript(sql)

DBHandler()