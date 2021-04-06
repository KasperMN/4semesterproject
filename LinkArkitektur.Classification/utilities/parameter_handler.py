from sklearn.model_selection import GridSearchCV
from typing import List

"""
    @ Hyper Parameter Handler
"""


class HyperParameterHandler:
    def find_hyperparameters_gb(self, model):
        # Creating dictionary
        hyper_parameters = dict(
            n_estimators=[10, 50, 100],
            learning_rate=[1, 3, 5],
            max_depth=[1, 2, 3])

        # Find best model
        return GridSearchCV(model, hyper_parameters, cv=3)

    def find_hyperparameters_knn(self, model):
        # Creating dictionary
        hyper_parameters = dict(
            leaf_size=[1, 5, 10],
            n_neighbors=[1, 5, 10],
            p=[1, 2, 3])

        # Find best model
        return GridSearchCV(model, hyper_parameters, cv=3)
