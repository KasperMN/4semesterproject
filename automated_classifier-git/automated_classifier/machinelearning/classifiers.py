from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class KNeighbors:
    def __init__(self):
        self._params = defaultdict(leaf_size=[1, 5, 10],
                                   n_neighbors=[1, 2, 3, 5, 10],
                                   p=[1, 2, 3])
        self._model = KNeighborsClassifier()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def find_best_estimator(self, training_features, training_labels):
        gscv = GridSearchCV(self._model, self._params, cv=3)
        gscv.fit(training_features, training_labels)
        self.model = gscv.best_estimator_


class GradientBoost:
    def __init__(self):
        self._params = defaultdict(n_estimators=[1, 10, 20],
                                   learning_rate=[1, 2, 3],
                                   max_depth=[1, 2])
        self._model = GradientBoostingClassifier()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def find_best_estimator(self, training_features, training_labels):
        gscv = GridSearchCV(self._model, self._params, cv=3)
        gscv.fit(training_features, training_labels)
        self._model = gscv.best_estimator_


class RandomForest:
    def __init__(self):
        self._params = defaultdict(n_estimators=[1, 10, 20],
                                   min_samples_split=[1, 2, 3],
                                   max_depth=[1, 2, 3],
                                   min_samples_leaf=[1, 2, 4])
        self._model = RandomForestClassifier()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def find_best_estimator(self, training_features, training_labels):
        rscv = RandomizedSearchCV(self._model, self._params, n_iter=50, cv=3)
        rscv.fit(training_features, training_labels)
        self._model = rscv.best_estimator_
