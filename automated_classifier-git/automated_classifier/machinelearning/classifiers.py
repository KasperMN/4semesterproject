import time
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class KNeighbors:
    def __init__(self, name):
        self._params = defaultdict(leaf_size=[1, 2, 3, 5, 10, 20, 50],
                                   n_neighbors=[1, 2, 3, 5, 10, 25],
                                   p=[1, 2, 3, 5, 10])
        self._model = KNeighborsClassifier()
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def find_best_estimator(self, training_features, training_labels, name):
        print("---> {}: Finding Parameters".format(name))
        start = time.time()

        gscv = GridSearchCV(self._model, self._params, cv=3)
        gscv.fit(training_features, training_labels.values.ravel())
        self.model = gscv.best_estimator_

        end = time.time()
        print('\n---> {0}: Found Parameters in {1:.2f} seconds'.format(name, end - start))
        print("Params: ", gscv.best_params_)
        return self


class GradientBoost:
    def __init__(self, name):
        self._params = defaultdict(n_estimators=[1, 2, 3, 4, 5, 10, 20],
                                   learning_rate=[1, 2, 3, 4, 5],
                                   max_depth=[1, 2, 3, 5])
        self._model = GradientBoostingClassifier()
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def find_best_estimator(self, training_features, training_labels, name):
        print("---> {}: Finding Parameters".format(name))
        start = time.time()

        gscv = GridSearchCV(self._model, self._params, cv=3)
        gscv.fit(training_features, training_labels.values.ravel())
        self._model = gscv.best_estimator_

        end = time.time()
        print('\n---> {0}: Found Parameters in {1:.2f} seconds '.format(name, end - start))
        print("Params: ", gscv.best_params_)
        return self


class RandomForest:
    def __init__(self, name):
        self._params = defaultdict(n_estimators=[1, 5, 10, 15, 20, 50, 100],
                                   min_samples_split=[1, 2, 3, 5, 10],
                                   max_depth=[1, 2, 3, 5, 10, 20, 40],
                                   min_samples_leaf=[1, 2, 3, 5, 10])
        self._model = RandomForestClassifier()
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def find_best_estimator(self, training_features, training_labels, name):
        print("---> {}: Finding Parameters".format(name))
        start = time.time()

        rscv = RandomizedSearchCV(self._model, self._params, n_iter=100, cv=3)
        rscv.fit(training_features, training_labels.values.ravel())
        self._model = rscv.best_estimator_

        end = time.time()
        print('\n---> {0}: Found Parameters in {1:.2f} seconds '.format(name, end - start))
        print("Params: ", rscv.best_params_)
        return self
