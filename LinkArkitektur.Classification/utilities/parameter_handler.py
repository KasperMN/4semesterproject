from sklearn.model_selection import GridSearchCV
from typing import List

"""
    @ Hyper Parameter Handler
"""


class HyperParameterHandler:
    def __init__(self):
        print("hi")

    def find_hyperparameters_gb(self, n_estimators: List[int], learning_rate: List[int], max_depth: List[int]):
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
        self.model = GridSearchCV(self.model, hyper_parameters, cv=3)

    def find_hyperparameters_knn(self, num_leaf_size: List[int], num_neighbors: List[int], num_p: List[int]):
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
        self.model = GridSearchCV(self.model, hyper_parameters, cv=3)
