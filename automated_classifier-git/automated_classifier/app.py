from __future__ import division
import os
import time
import pandas as pd
from pandas import DataFrame
from automated_classifier import data as dt
from automated_classifier import machinelearning as ml
import joblib
import hickle as hkl


class TrainerApp:
    def __init__(self, link: str = "", columns: list = None, target: str = "", table_name: str = ""):
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

        # test_query = "select * from walls limit 100"

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


class PredictApp:
    def __init__(self):
        self._filename = "program/trained_model.sav"

    def predict_classification(self, data: DataFrame):
        # Load the model
        loaded_model = joblib.load(filename=self._filename)

        # Process data before prediction
        le = hkl.load('program/le.hkl')
        qt = hkl.load('program/qt.hkl')

        p = dt.PreProcessing(data)
        pred_num = p.get_numerical_columns(data)
        pred_cat = p.get_categorical_columns(data)

        data.loc[:, pred_cat] = le.transform(data.loc[:, pred_cat].values.ravel())
        data.loc[:, pred_num] = qt.transform(data.loc[:, pred_num])

        assemblycodes = hkl.load('program/assemblycodes.hkl')

        # Make predictions
        probabilities = loaded_model.predict_proba(data)
        prediction = loaded_model.predict(data)

        res = ""

        for i, _ in enumerate(prediction):
            res = {"Prediction": assemblycodes[str(prediction[i])], "Percentage": max(probabilities[i])}

        return res

    def test_predict_dataset(self):
        # Load data
        data = pd.read_csv("data/test1.csv", index_col=0)

        # Load model and make predict and probability
        loaded_model = joblib.load('program/trained_model.sav')
        probabilities = loaded_model.predict_proba(data)
        prediction = loaded_model.predict(data)

        assemblycodes = hkl.load('program/assemblycodes.hkl')

        # Load encoder and transformer to inverse the dataset
        le = hkl.load('program/le.hkl')
        qt = hkl.load('program/qt.hkl')

        numcols = data.drop('Base Constraint', axis=1)
        numcols_inversed = qt.inverse_transform(numcols)
        df = pd.DataFrame(numcols_inversed, columns=numcols.columns)
        df['Base Constraint'] = le.inverse_transform(data['Base Constraint'])

        pred = []
        prob = []

        for i,_ in enumerate(prediction):
            pred.append(format(assemblycodes[str(prediction[i])]))
            prob.append(max(probabilities[i]))

        df['Assembly Code'] = pred
        df['Probability'] = prob
        df.to_csv("predtest.csv")


if __name__ == '__main__':
    import pandas as pd
    app = TrainerApp("", ["Area","Base Constraint","Length","Structural Usage","Unconnected Height","Volume","Assembly Code","Width"],
                     'Assembly Code', 'table')
    app._selected_data = pd.read_csv('data/combineddata.csv')

    app.insert_data_to_sqllite()
    app.process_data_from_db()
    app.create_and_fit_models()
    app.make_accuracy_test()
    app.save_best_model()
