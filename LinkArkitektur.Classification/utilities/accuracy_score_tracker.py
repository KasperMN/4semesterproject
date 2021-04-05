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
        # NAMES OF TESTED CLASSIFIERS
        self.accuracy_filename = "C:\\LinkArkitektur\\Accuracy_Scores_Library.hkl"
        self.index = []

        # TYPE OF SCORING TO SAVE
        self.names_filename = "C:\\LinkArkitektur\\Classifier_Names_Library.hkl"
        self.scores = {"Accuracy": [], "Balanced Accuracy": [], "Weighted Accuracy": []}
        self.scoring = ["accuracy", "balanced_accuracy", "f1_weighted"]

        # FEATURES & LABELS
        self.test_features = test_features
        self.test_labels = test_labels

        # DATAFRAME SCORES
        self.df_scores = ''

        # TRY TO LOAD FILES IF EXISTING
        try:
            self.scores = hkl.load(self.accuracy_filename)
            self.index = hkl.load(self.names_filename)
            #print(self.scores)
            #print(self.index)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
        except:  # handle other exceptions such as attribute errors
            print("Unexpected error:", sys.exc_info()[0])
        #print("done")

    def add_score(self, name, classifier):
        if name not in self.index:
            # CROSS VALIDATION
            cv_result = cross_validate(classifier, X=self.test_features, y=self.test_labels, scoring=self.scoring)
            self.scores["Accuracy"].append(cv_result["test_accuracy"].mean())
            self.scores["Balanced Accuracy"].append(cv_result["test_balanced_accuracy"].mean())
            self.scores["Weighted Accuracy"].append(cv_result["test_f1_weighted"].mean())
            self.index.append(name)

        else:
            print("Classifier already added: {}".format(name))

        # CREATE DATAFRAME
        self.df_scores = pd.DataFrame(self.scores, index=self.index)

        # SAVE FILES WITH NEW SCORES ADDED
        hkl.dump(self.scores, self.accuracy_filename)
        hkl.dump(self.index, self.names_filename)

    def display_scores(self):
        print("\n")
        print("\n@@ Accuracy Scores @@")
        # PRINT RESULT
        print(self.df_scores)
