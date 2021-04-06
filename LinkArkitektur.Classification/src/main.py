from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
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

""" @ Create models """
knn_model = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1)
gb_model = GradientBoostingClassifier(learning_rate=1, max_depth=2, n_estimators=10)
rf_model = RandomForestClassifier(n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_depth=50)

knn_model_smote = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1)
gb_model_smote = GradientBoostingClassifier(learning_rate=1, max_depth=3, n_estimators=10)
rf_model_smote = RandomForestClassifier(n_estimators=200, min_samples_split=2, min_samples_leaf=
                                        4, max_depth=110)
""" @ Find hyperparameters """
fhp = utilities.HyperParameterHandler()  # fhp = Find Hyper Parameters
#knn_model = fhp.find_hyperparameters_knn(model=knn_model)
#gb_model = fhp.find_hyperparameters_gb(model=gb_model)
#knn_model_smote = fhp.find_hyperparameters_knn(model= knn_model_smote)
#gb_model_smote = fhp.find_hyperparameters_gb(model= gb_model_smote)
#rf_model = fhp.find_hyperparameters_rf(model=rf_model)
#rf_model_smote = fhp.find_hyperparameters_rf(model=rf_model_smote)

""" @ Train models """
#knn_model.fit(training_features, training_labels)  # Training KNN Model
#print("Knn Model:", knn_model.best_params_)
#gb_model.fit(training_features, training_labels)  # Training GB Model
#print("Gb Model:", gb_model.best_params_)
#knn_model_smote.fit(training_features_smote, training_labels_smote)  # Training KNN SMOTE Model
#print("Knn Smote:", knn_model_smote.best_params_)
#gb_model_smote.fit(training_features_smote,training_labels_smote)  # Training GB SMOTE Model
#print("Gb Smote:", gb_model_smote.best_params_)
rf_model.fit(training_features,training_labels)  # Training GB SMOTE Model
#print("Rf model:", rf_model.best_params_)
rf_model_smote.fit(training_features_smote,training_labels_smote)  # Training GB SMOTE Model
#print("Rf Smote:", rf_model_smote.best_params_)

""" @ Test accuracy score """
at = utilities.AccuracyTracker(test_features=test_features, test_labels=test_labels)

at.add_score(name='KNeighbors Classifier - Original', classifier=knn_model)
at.add_score(name='KNeighbors Classifier - Oversampled', classifier=knn_model_smote)
at.add_score(name='GradientBoosting Classifier - Original', classifier=gb_model)
at.add_score(name='GradientBoosting Classifier Oversampled', classifier=gb_model_smote)
at.add_score(name='RandomForestClassifier - Original', classifier=rf_model)
at.add_score(name='RandomForestClassifier - Oversampled', classifier=rf_model_smote)

""" @ Display scores """
at.display_scores()