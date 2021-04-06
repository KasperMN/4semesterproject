from sklearn.model_selection import cross_validate
from pandas import DataFrame
import hickle as hkl
import pandas as pd
import sys

# Settings if printing dataframe
from typing import List

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 20)


class AccuracyTracker:
    def __init__(self, test_features: DataFrame, test_labels: DataFrame):
        self.index = []
        self.scores = {"Accuracy": [], "Balanced Accuracy": [], "Weighted Accuracy": []}
        self.scoring = ["accuracy", "balanced_accuracy", "f1_weighted"]
        self.test_features = test_features
        self.test_labels = test_labels
        self.df_scores = ''  # DATAFRAME SCORES
        self.best_classifier = {}
        self.best_score = 0

    def save_best_model(self):
        hkl.dump(self.best_classifier, "..\\..\\Data\\Best_Classifier.hkl")
        print("\nBest Classifier: ", self.best_classifier.get('classifier'))
        print("Accuracy: {0:.2f} %".format(self.best_classifier.get('accuracy') * 100))
        print("Balanced Accuracy: {0:.2f} %".format(self.best_classifier.get('balanced_accuracy') * 100))
        print("Weighted Accuracy: {0:.2f} %".format(self.best_classifier.get('f1_weighted') * 100))

    def add_score(self, name, classifier):
        cv_result = cross_validate(classifier, X=self.test_features, y=self.test_labels, scoring=self.scoring)
        self.index.append(name)
        self.scores["Accuracy"].append("{0:.2f} %".format(cv_result["test_accuracy"].mean() * 100))
        self.scores["Balanced Accuracy"].append("{0:.2f} %".format(cv_result["test_balanced_accuracy"].mean() * 100))
        self.scores["Weighted Accuracy"].append("{0:.2f} %".format(cv_result["test_f1_weighted"].mean() * 100))
        self.df_scores = pd.DataFrame(self.scores, index=self.index)  # CREATE DATAFRAME
        score = cv_result["test_accuracy"].mean() + \
                cv_result["test_balanced_accuracy"].mean() + \
                cv_result["test_f1_weighted"].mean()

        if self.best_score < score:
            self.best_score = score
            self.best_classifier = dict(
                name=name,
                classifier=classifier,
                accuracy=cv_result["test_accuracy"].mean(),
                balanced_accuracy=cv_result["test_balanced_accuracy"].mean(),
                f1_weighted=cv_result["test_f1_weighted"].mean())

    def display_scores(self):
        print("@@ Accuracy Scores @@")
        # PRINT RESULT
        print(self.df_scores)



"""  try:  # TRY TO LOAD FILES IF EXISTING
            current_best_model = hkl.load("..\\..\\Data\\Best_Classifier.hkl")
            self.index = hkl.load(self.names_filename)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
        except:  # handle other exceptions such as attribute errors
            print("Unexpected error:", sys.exc_info()[0])"""