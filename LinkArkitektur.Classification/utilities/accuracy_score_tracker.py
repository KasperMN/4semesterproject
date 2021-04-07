from sklearn.model_selection import cross_validate
from pandas import DataFrame
import numpy as np
import hickle as hkl
import pandas as pd

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 20)


class AccuracyTracker:
    def __init__(self, test_features: DataFrame, test_labels: DataFrame):
        self.scoring = ["accuracy", "balanced_accuracy", "f1_weighted"]
        self.test_features = test_features
        self.test_labels = test_labels
        self.scores = {}
        self.index = []
        self.df_scores = DataFrame  # For displaying scores
        self.best_classifier = {}  # Best Classifiers stats
        self.best_score = []

    def add_score(self, name, classifier):
        self.index.append(name)
        cv_result = cross_validate(
            estimator=classifier, X=self.test_features, y=self.test_labels, scoring=self.scoring,
            verbose=1, n_jobs=1, cv=10)

        for _, element in enumerate(cv_result):
            if element not in self.scores:
                self.scores[element] = []
            if "test" in element:
                score = self.scores[element].append("{0:.2f} %".format(cv_result[element].mean() * 100))
            elif "time" in element:
                self.scores[element].append("{0:.2f} ms".format(cv_result[element].mean() * 1000))

    def display_scores(self):
        self.df_scores = pd.DataFrame(self.scores, index=self.index)  # CREATE DATAFRAME
        self.df_scores = self.df_scores.sort_values(by=['test_accuracy', 'test_balanced_accuracy', 'test_f1_weighted'], ascending=False)
        print("@@ Accuracy Scores @@ \n {}".format(self.df_scores))
