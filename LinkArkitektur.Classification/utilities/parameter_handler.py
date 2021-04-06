from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from typing import List

"""
    @ Hyper Parameter Handler
"""


class HyperParameterHandler:
    def find_hyperparameters_gb(model, n_estimators: List[int], learning_rate: List[int], max_depth: List[int]):
        # Listing ranges to test
        range_n_estimators = n_estimators
        range_learning_rate = learning_rate
        range_max_depth = max_depth

        # Creating dictionary
        hyper_parameters = dict(
            n_estimators=range_n_estimators,
            learning_rate=range_learning_rate,
            max_depth=range_max_depth)

        # Find best model
        return GridSearchCV(model, hyper_parameters, cv=3)

    def find_hyperparameters_knn(model, num_leaf_size: List[int], num_neighbors: List[int], num_p: List[int]):
        # Listing ranges to test
        range_leaf_size = num_leaf_size
        range_n_neighbors = num_neighbors
        range_p = num_p

        # Creating dictionary
        hyper_parameters = dict(
            leaf_size=range_leaf_size,
            n_neighbors=range_n_neighbors,
            p=range_p)

        # Find best model
        return GridSearchCV(model, hyper_parameters, cv=3)

    def find_hyperparameters_rf(self, model):

        hyper_parameters = dict(
            n_estimators=[int(x) for x in np.linspace(start=200, stop=2000, num=10)],
            max_depth=[int(x) for x in np.linspace(10, 110, num=11)],
            min_samples_split=[2, 5, 10],
            min_samples_leaf=[1, 2, 4]
        )

        return RandomizedSearchCV(model, hyper_parameters, n_iter=100, cv=3)
