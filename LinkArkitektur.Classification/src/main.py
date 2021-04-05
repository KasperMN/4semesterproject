from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import pandas
import warnings
warnings.filterwarnings("ignore")

import utilities

# CREATE DATAHANDLER
csv_handler = utilities.CsvHandler()
data_handler = utilities.DataHandler()
plot_handler = utilities.PlotHandler()

# COLLECT THE DATA
data, target, features = csv_handler.get_csv_data(
    file_route="..\\..\\Data\\modified_data.csv",
    file_separator=",",
    target_column="Assembly_Code",
    tabels_to_drop=["Type_Id", "Assembly_Code"])

"""
    @ Preprocess data
"""
training_features, test_features, training_labels, test_labels = data_handler.preprocessing_data(
    features=features, target=target)

"""
    @ Oversampled data
"""
oversample = SMOTE(k_neighbors=1)
training_features_smote, training_labels_smote = oversample.fit_resample(training_features, training_labels)
training_features_smote.to_csv()
"""
    @ Create models
"""
knn_model = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1)
gb_model = GradientBoostingClassifier(learning_rate=1, max_depth=2, n_estimators=10)

knn_model_smote = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1)
gb_model_smote = GradientBoostingClassifier(learning_rate=1, max_depth=3, n_estimators=10)
"""
    @ Find hyperparameters
"""
#fhp.find_hyperparameters_gb(n_estimators=[10, 50, 100], learning_rate=[1, 3, 5], max_depth=[1, 2, 3])
#fhp.find_hyperparameters_knn(num_leaf_size=[1, 5, 10], num_neighbors=[1, 5, 10], num_p=[1, 2, 3])

#knn_model_smote = utilities.HyperParameterHandler.find_hyperparameters_knn(knn_model_smote, num_leaf_size=[1, 5, 10], num_neighbors=[1, 5, 10], num_p=[1, 2, 3])

#gb_model_smote = utilities.HyperParameterHandler.find_hyperparameters_gb(gb_model_smote, n_estimators=[10, 50, 100], learning_rate=[1, 3, 5], max_depth=[1, 2, 3])


"""
    @ Train models
"""
knn_model.fit(training_features, training_labels)
gb_model.fit(training_features, training_labels)

knn_model_smote.fit(training_features_smote, training_labels_smote)
gb_model_smote.fit(training_features_smote,training_labels_smote)

"""
    @ Test accuracy score
"""
at = utilities.AccuracyTracker(test_features=test_features, test_labels=test_labels)

at.add_score(name='KNeighborsClassifier', classifier=knn_model)
at.add_score(name='GradientBoostingClassifier', classifier=gb_model)
at.add_score(name='KNN smote', classifier=knn_model_smote)
at.add_score(name='GBC Smote', classifier=gb_model_smote)
"""
    @ Smote accuracy test
"""

"""
    @ Display scores
"""
at.display_scores()