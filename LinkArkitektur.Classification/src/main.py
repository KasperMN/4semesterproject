from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import hickle as hkl
import warnings
import utilities
warnings.filterwarnings("ignore")

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
rf_model = RandomForestClassifier()
nb_model = MultinomialNB()

''' @ Create SMOTE Models'''
knn_model_smote = KNeighborsClassifier(leaf_size=1, n_neighbors=10, p=1)
gb_model_smote = GradientBoostingClassifier(learning_rate=2, max_depth=3, n_estimators=1)
rf_model_smote = RandomForestClassifier()
nb_model_smote = MultinomialNB()

''' @ Find Hyper Parameters for models '''
'''
fhp = utilities.HyperParameterHandler()  # fhp = Find Hyper Parameters
knn_model = fhp.find_hyperparameters_knn(model=knn_model)  # Returns GridSearchCV Object
gb_model = fhp.find_hyperparameters_gb(model=gb_model)  # Returns GridSearchCV Object
knn_model_smote = fhp.find_hyperparameters_knn(model= knn_model_smote)  # Returns GridSearchCV Object
gb_model_smote = fhp.find_hyperparameters_gb(model= gb_model_smote)  # Returns GridSearchCV Object
'''

''' @ Train Original Models '''
knn_model.fit(training_features, training_labels)  # Training KNN Model
gb_model.fit(training_features, training_labels)  # Training GB Model
nb_model.fit(training_features, training_labels)  # Training NB Model

''' @ Train SMOTE Models '''
knn_model_smote.fit(training_features_smote, training_labels_smote)  # Training KNN SMOTE Model
gb_model_smote.fit(training_features_smote,training_labels_smote)  # Training GB SMOTE Model
nb_model_smote.fit(training_features_smote, training_labels_smote)  # Training NB SMOTE Model

''' @ Print Best Parameters '''
'''
print("Knn Model:", knn_model.best_params_)
print("Gb Model:", gb_model.best_params_)
print("Knn Smote:", knn_model_smote.best_params_)
print("Gb Smote:", gb_model_smote.best_params_)
'''

''' @ Test accuracy score '''
at = utilities.AccuracyTracker(test_features=test_features, test_labels=test_labels)
at.add_score(name='KNeighbors - Original', classifier=knn_model)
at.add_score(name='KNeighbors - Oversampled', classifier=knn_model_smote)
at.add_score(name='GradientBoosting - Original', classifier=gb_model)
at.add_score(name='GradientBoosting Oversampled', classifier=gb_model_smote)
at.add_score(name='RandomForest - Original', classifier=rf_model)
at.add_score(name='RandomForest - Oversampled', classifier=rf_model_smote)
at.add_score(name='NaiveBayes - Original', classifier=nb_model)
at.add_score(name='NaiveBayes - Oversampled', classifier=nb_model_smote)

''' @ Display scores '''
at.display_scores()

''' @ Save Best Model '''
at.save_best_model()
