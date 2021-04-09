from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


class HyperParameterHandler:
    def find_hyperparameters_gb(self, model):
        hyper_parameters = dict(   # Creating dictionary with variables
            n_estimators=[1, 10, 20, 50],
            learning_rate=[1, 2, 3, 5],
            max_depth=[1, 2, 3])

        return GridSearchCV(model, hyper_parameters, cv=3)

    def find_hyperparameters_knn(self, model):
        hyper_parameters = dict(  # Creating dictionary with variables
            leaf_size=[1, 5, 10],
            n_neighbors=[1, 2, 3, 5, 10],
            p=[1, 2, 3])

        return GridSearchCV(model, hyper_parameters, cv=3)

    def find_hyperparameters_rf(self, model):
        hyper_parameters = dict(  # Creating dictionary with variables
            n_estimators=[int(x) for x in np.linspace(start=200, stop=2000, num=10)],
            max_depth=[int(x) for x in np.linspace(10, 110, num=11)],
            min_samples_split=[2, 5, 10],
            min_samples_leaf=[1, 2, 4])

        return RandomizedSearchCV(model, hyper_parameters, n_iter=100, cv=3)
