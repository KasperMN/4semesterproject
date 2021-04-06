from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import warnings
import utilities
warnings.filterwarnings("ignore")

""" @ Collect data from CSV files in GitHub Repository """
training_features = pd.read_csv(r'..\..\Data\training_features.csv', sep=',')
test_features = pd.read_csv(r'..\..\Data\test_features.csv', sep=',')
training_labels = pd.read_csv(r'..\..\Data\training_labels.csv', sep=',')
test_labels = pd.read_csv(r'..\..\Data\test_labels.csv', sep=',')

training_features_smote = pd.read_csv(r'..\..\Data\training_features_smote.csv')
training_labels_smote = pd.read_csv(r'..\..\Data\training_labels_smote.csv')

""" @ Create models """
knn_model = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1)
gb_model = GradientBoostingClassifier(learning_rate=1, max_depth=2, n_estimators=10)
rf_model = RandomForestClassifier()

knn_model_smote = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1)
gb_model_smote = GradientBoostingClassifier(learning_rate=1, max_depth=3, n_estimators=10)
rf_model_smote = RandomForestClassifier()
""" @ Find hyperparameters """
#fhp.find_hyperparameters_gb(n_estimators=[10, 50, 100], learning_rate=[1, 3, 5], max_depth=[1, 2, 3])
#fhp.find_hyperparameters_knn(num_leaf_size=[1, 5, 10], num_neighbors=[1, 5, 10], num_p=[1, 2, 3])
#knn_model_smote = utilities.HyperParameterHandler.find_hyperparameters_knn(knn_model_smote, num_leaf_size=[1, 5, 10], num_neighbors=[1, 5, 10], num_p=[1, 2, 3])
#gb_model_smote = utilities.HyperParameterHandler.find_hyperparameters_gb(gb_model_smote, n_estimators=[10, 50, 100], learning_rate=[1, 3, 5], max_depth=[1, 2, 3])


""" @ Train models """
knn_model.fit(training_features, training_labels)
gb_model.fit(training_features, training_labels)
rf_model.fit(training_features, training_labels)

knn_model_smote.fit(training_features_smote, training_labels_smote)
gb_model_smote.fit(training_features_smote,training_labels_smote)
rf_model_smote.fit(training_features_smote, training_labels_smote)

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