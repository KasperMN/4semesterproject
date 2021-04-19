from __future__ import division
import os
import time
from automated_classifier import data as dt
from automated_classifier import machinelearning as ml
import joblib


class Application:
    def __init__(self, link: str, columns: list, target: str, table_name: str):
        self._db_connection = dt.Connection(table_name, columns)
        self._data_collector = dt.DataCollector(chosen_columns=columns)
        self._selected_data = None
        self._api_link = link
        self._target = target
        self._table_name = table_name
        self._fitted_models = None
        self._processed_data = None
        self._best_model_name = ""
        self._best_model = None

    def return_best_model(self):
        start = time.time()

        # Collect Data From External Api
        self.collect_data_from_api()

        # Collect chosen columns from api data
        self.collect_columns_from_data()

        if self._selected_data is None:
            return self._data_collector.error_messages

        # Create DB and insert data
        self.insert_data_to_sqllite()

        # Pre process the data
        self.process_data_from_db()

        # Create the models
        self.create_and_fit_models()

        # Test each model for accuracy and save best model
        self.make_accuracy_test()

        # Save the best model to folder "program"
        self.save_best_model()

        end = time.time()
        print('@@ Found Best Model in {:.2f} minutes'.format((end - start) / 60))
        return {"Model Name": self._best_model_name, 'Model Score': self._best_model_score}

    def predict_classification(self):
        # Load the model
        # filename = "program/trained_model.sav"
        # loaded_model = joblib.load(filename=filename)
        # result = loaded_model.score(data)
        # Make predictions
        # Return the predictions as file or strings
        return None

    def collect_data_from_api(self):
        print("\n@@ Collect data from external API")
        start = time.time()

        self._data_collector.api_data = self._api_link

        end = time.time()
        print('---> @ Data collected in {:.2f} seconds'.format(end - start))

    def collect_columns_from_data(self):
        print("\n@@ Collect chosen columns from data")
        start = time.time()

        self._selected_data = self._data_collector.chosen_data

        end = time.time()
        print('---> @ Columns collected in {:.2f} seconds'.format(end - start))

    def insert_data_to_sqllite(self):
        print("\n@@ Create Database and Insert Data Into SqlLite-Database")
        start = time.time()

        self._db_connection.create_database()  # Create the connection and database
        self._db_connection.insert_data(self._selected_data)  # Insert the data

        end = time.time()
        print('---> @ Data inserted in {:.2f} seconds'.format(end - start))

    def process_data_from_db(self):
        print("\n@@ Process the Data")
        start = time.time()

        self._db_connection.create_select_query()
        unprocessed_data = self._db_connection.get_data()
        pre_processor = dt.PreProcessing(unprocessed_data)
        pre_processor.create_processed_data(self._target)
        self._processed_data = pre_processor.processed_data

        end = time.time()
        print('---> @ Data processed in {:.2f} seconds'.format(end - start))

    def create_and_fit_models(self):
        print("\n@@ Create Models and Fit them")
        start = time.time()

        model_handler = ml.ModelHandler(self._processed_data)
        model_handler.create_models()
        model_handler.fit_models()
        self._fitted_models = model_handler.fitted_models

        end = time.time()
        print('---> @ Data processed in {:.2f} seconds'.format(end - start))

    def make_accuracy_test(self):
        accuracy_handler \
            = ml.AccuracyHandler(self._processed_data["test_features"], self._processed_data["test_labels"])
        for key, model in self._fitted_models.items():
            accuracy_handler.add_score(key, model)
        accuracy_handler.display_scores()
        self._best_model_score = accuracy_handler.best_model_score
        self._best_model_name = accuracy_handler.best_model_name
        self._best_model = self._fitted_models[self._best_model_name]

    def save_best_model(self):
        filename = "program/trained_model.sav"
        if os.path.exists(r"program/trained_model.sav"):
            os.remove(r"program/trained_model.sav")
        joblib.dump(self._best_model, filename)


