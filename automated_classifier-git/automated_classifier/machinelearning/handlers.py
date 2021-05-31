import concurrent.futures
import threading

from automated_classifier.machinelearning import classifiers
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import cross_validate
import multiprocessing as mp


class ModelHandler:
    def __init__(self, data: dict):
        self._data = data
        self._org_models = {}
        self._fitted_models = {}

    @property
    def models(self):
        return self._org_models

    @property
    def fitted_models(self):
        return self._fitted_models

    def create_models(self):
        self._org_models["KNeighbors"] = classifiers.KNeighbors("KNeighbors")
        self._org_models["GradientBoost"] = classifiers.GradientBoost("GradientBoost")
        self._org_models["RandomForest"] = classifiers.RandomForest("RandomForest")
        self._org_models["SupportVector"] = classifiers.SupportVector("SupportVector")
        self._org_models["KNeighbors_Smote"] = classifiers.KNeighbors("KNeighbors_Smote")
        self._org_models["GradientBoost_Smote"] = classifiers.GradientBoost("GradientBoost_Smote")
        self._org_models["RandomForest_Smote"] = classifiers.RandomForest("RandomForest_Smote")
        self._org_models["SupportVector_Smote"] = classifiers.SupportVector("SupportVector_Smote")

    def insert_model(self, result):
        self._fitted_models[result.name] = result.model

    def fit_models(self):
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = []
            for key, classifier in self._org_models.items():
                name = classifier.name
                results.append(executor.submit(classifier.find_best_estimator,
                                               self._data["training_features"], self._data["training_labels"], name))

            for res in results:
                self.insert_model(res.result())

        """

        # Step 1: Init multiprocessing.Pool()
        pool = mp.Pool(mp.cpu_count() - 1)

        # Step 2: pool.apply
        for key, classifier in self._org_models.items():
            name = classifier.name
            pool.apply_async(
                classifier.find_best_estimator, args=(
                    self._data["training_features"], self._data["training_labels"], name), callback=self.insert_model)

        pool.close()
        pool.join()

class AccuracyHandler:
    def __init__(self, test_features: DataFrame, test_labels: DataFrame):
        self.scoring = ["accuracy", "balanced_accuracy", "f1_weighted"]  # Type of accuracies we want
        self.test_features = test_features  # Features for the test set
        self.test_labels = test_labels  # Labels for the test set
        self.scores = {'total_accuracy': []}  # Collecting the scores
        self.index = []  # Names for classifiers
        self.total_accuracy = 0  # For incrementing accuracy
        self.df_scores = DataFrame  # For displaying score
        self._best_model_name = None
        self._best_model_score = ""

    @property
    def best_model_name(self):
        self._best_model_name = self.df_scores.index[0]
        return self._best_model_name

    @property
    def best_model_score(self):
        self._best_model_score = self.df_scores.iloc[0]
        return self._best_model_score.to_dict()

    def add_score(self, name, classifier):
        self.index.append(name)
        cv_result = cross_validate(  # Iterating over 10 pieces of data to find best score
            estimator=classifier, X=self.test_features, y=self.test_labels.values.ravel(), scoring=self.scoring,
            verbose=0, n_jobs=2, cv=6)
        for _, element in enumerate(cv_result):  # element is column name
            if element not in self.scores:  # Add new column if it does not exist
                self.scores[element] = []  # Creates new column with that name
            if "test" in element:  # 'test' is included in accuracy scores
                self.scores[element].append("{:.2f} %".format(cv_result[element].mean() * 100))
                self.total_accuracy += cv_result[element].mean() * 100
            elif "time" in element:  # 'time' is included in the time measures
                self.scores[element].append("{:.0f} ms".format(cv_result[element].mean() * 1000))
        self.scores['total_accuracy'].append("{:.2f}".format(self.total_accuracy))
        self.total_accuracy = 0  # Resetting the score, because all use same instance

    def display_scores(self):
        self.df_scores = pd.DataFrame(self.scores, index=self.index)  # CREATE DATAFRAME
        self.df_scores = self.df_scores.sort_values(by=['total_accuracy'], ascending=False)
        print("\n@@ Accuracy Scores @@ \n {}".format(self.df_scores))

    def get_score(self):
        self.df_scores = pd.DataFrame(self.scores, index=self.index)  # CREATE DATAFRAME
        self.df_scores = self.df_scores.sort_values(by=['total_accuracy'], ascending=False)
        return self.df_scores
