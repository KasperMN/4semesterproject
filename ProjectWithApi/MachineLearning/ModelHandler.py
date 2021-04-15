import os
import sqlite3
from collections import defaultdict
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import warnings
import MachineLearning.PreProcessing

warnings.filterwarnings("ignore")


class ModelHandler:
    def __init__(self, data: list):
        self.training_features = data[0]
        self.training_labels = data[1]
        self.training_features_smote = data[2]
        self.training_labels_smote = data[3]
        self.test_features = data[4]
        self.test_labels = data[5]


    def create_models(self):
        ''' @ Create Original Models '''
        knn_model = KNeighborsClassifier()
        gb_model = GradientBoostingClassifier()
        rf_model = RandomForestClassifier()

        return [knn_model, gb_model]

    def create_smote_models(self):
        ''' @ Create SMOTE Models'''
        knn_model_smote = KNeighborsClassifier()
        gb_model_smote = GradientBoostingClassifier()
        rf_model_smote = RandomForestClassifier()

        return [knn_model_smote, gb_model_smote]

    def train_models(self, models: list):
        trained_models = []
        for model in models:
            trained_models.append(model.fit(self.training_features, self.training_labels))
        return trained_models

    def train_smote_models(self, models: list):
        ''' @ Train SMOTE Models '''
        trained_models = []
        for model in models:
            trained_models.append(model.fit(self.training_features_smote, self.training_labels_smote))
        return trained_models

    def return_model_cv(self, model):
        knn_param = defaultdict(leaf_size=[1, 5, 10],n_neighbors=[1, 2, 3, 5, 10],p=[1, 2, 3])
        gb_param = defaultdict(n_estimators=[1, 10, 20, 50],learning_rate=[1, 2, 3, 5],max_depth=[1, 2, 3])
        rf_param = defaultdict(n_estimators=[int(x) for x in np.linspace(start=200, stop=1000, num=100)],
                               max_depth=[int(x) for x in np.linspace(10, 110, num=11)],
                               min_samples_split=[2, 5, 10],
                               min_samples_leaf=[1, 2, 4])

        if type(model) == KNeighborsClassifier:
            return GridSearchCV(model, knn_param, cv=3)
        elif type(model) == GradientBoostingClassifier:
            return GridSearchCV(model, gb_param, cv=3)
        elif type(model) == RandomForestClassifier:
            return RandomizedSearchCV(model, rf_param, n_iter=100, cv=3)



if __name__ == '__main__':
    conn = sqlite3.connect(r"../DataBase/database.db")
    df = pd.read_sql("SELECT [Area], [Volume], [Assembly Code] FROM [walls]", conn)
    p = MachineLearning.PreProcessing(df)
    p.returns_processed_test_and_training_data("Assembly Code")
    mh = ModelHandler(p.get_data())
    original_models = mh.create_models()
    smote_models = mh.create_smote_models()

    l = [mh.return_model_cv(model) for model in original_models]
    mh.train_models(l)

    l2 = [mh.return_model_cv(model) for model in smote_models]
    mh.train_smote_models(l2)



