from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import warnings
import utilities
warnings.filterwarnings("ignore")


class Application:
    def __init__(self):
        """ @ Collect data from CSV files in GitHub Repository """
        training_features = pd.read_csv(r'..\..\Data\training_features.csv', sep=',')  # Training features
        test_features = pd.read_csv(r'..\..\Data\test_features.csv', sep=',')  # Testing features
        training_labels = pd.read_csv(r'..\..\Data\training_labels.csv', sep=',')  # Training labels
        test_labels = pd.read_csv(r'..\..\Data\test_labels.csv', sep=',')  # Testing labels
        training_features_smote = pd.read_csv(r'..\..\Data\training_features_smote.csv')  # Oversampled data
        training_labels_smote = pd.read_csv(r'..\..\Data\training_labels_smote.csv')  # Oversampled data

        ''' @ Create Original Models '''
        knn_model = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1)
        gb_model = GradientBoostingClassifier(learning_rate=1, max_depth=3, n_estimators=1)
        rf_model = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_depth=50)

        ''' @ Create SMOTE Models'''
        knn_model_smote = KNeighborsClassifier(leaf_size=1, n_neighbors=10, p=1)
        gb_model_smote = GradientBoostingClassifier(learning_rate=2, max_depth=3, n_estimators=1)
        rf_model_smote = RandomForestClassifier(n_estimators=200, min_samples_split=2, min_samples_leaf=4, max_depth=110)

        ''' @ Train Original Models '''
        knn_model.fit(training_features, training_labels)  # Training KNN Model
        gb_model.fit(training_features, training_labels)  # Training GB Model
        rf_model.fit(training_features,training_labels)  # Training GB SMOTE Model

        ''' @ Train SMOTE Models '''
        knn_model_smote.fit(training_features_smote, training_labels_smote)  # Training KNN SMOTE Model
        gb_model_smote.fit(training_features_smote,training_labels_smote)  # Training GB SMOTE Model
        rf_model_smote.fit(training_features_smote,training_labels_smote)  # Training GB SMOTE Model

        ''' @ Test accuracy score '''
        at = utilities.AccuracyTracker(test_features=test_features, test_labels=test_labels)  # Instance to handle accuracy
        at.add_score(name='KNeighbors - Original', classifier=knn_model)  # Testing: KNeighbors - Original
        at.add_score(name='KNeighbors - Oversampled', classifier=knn_model_smote)  # Testing: KNeighbors - Oversampled
        at.add_score(name='GradientBoosting - Original', classifier=gb_model)  # Testing: GradientBoosting - Original
        at.add_score(name='GradientBoosting Oversampled', classifier=gb_model_smote)  # Testing: GradientBoosting Oversampled
        at.add_score(name='RandomForest - Original', classifier=rf_model)  # Testing: RandomForest - Original
        at.add_score(name='RandomForest - Oversampled', classifier=rf_model_smote)  # Testing: RandomForest - Oversampled

        ''' @ Display scores '''
        #at.display_scores()
        self.data = at.get_score()

    def get_data(self):
        return self.data
