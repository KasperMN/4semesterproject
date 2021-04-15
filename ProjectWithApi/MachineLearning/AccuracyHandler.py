from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import cross_validate


class AccuracyHandler:
    def __init__(self, test_features: DataFrame, test_labels: DataFrame):
        self.scoring = ["accuracy", "balanced_accuracy", "f1_weighted"]  # Type of accuracies we want
        self.test_features = test_features  # Features for the test set
        self.test_labels = test_labels  # Labels for the test set
        self.scores = {'total_accuracy': []}  # Collecting the scores
        self.index = []  # Names for classifiers
        self.total_accuracy = 0  # For incrementing accuracy
        self.df_scores = DataFrame  # For displaying scores

    def add_score(self, name, classifier):
        self.index.append(name)
        cv_result = cross_validate(  # Iterating over 10 pieces of data to find best score
            estimator=classifier, X=self.test_features, y=self.test_labels, scoring=self.scoring,
            verbose=1, n_jobs=1, cv=10)

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
        print("@@ Accuracy Scores @@ \n {}".format(self.df_scores))

    def get_score(self):
        self.df_scores = pd.DataFrame(self.scores, index=self.index)  # CREATE DATAFRAME
        self.df_scores = self.df_scores.sort_values(by=['total_accuracy'], ascending=False)
        return self.df_scores
