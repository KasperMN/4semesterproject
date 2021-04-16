from __future__ import division
import sys

import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 20)


class PreProcessing:
    def __init__(self, data: DataFrame):
        self._data = data
        self._processed_data = {}

    def create_processed_data(self, target: str):
        print('Shape of Data: {}'.format(self._data.shape[0]))  # Prints num rows

        ''' @@ Remove Duplicates From Data '''
        self._data.drop_duplicates(inplace=True)  # Drops duplicate rows
        print('Shape of Data after dropping duplicates: {}'.format(self._data.shape[0]))  # Prints num rows

        hej = self._data[target].value_counts()
        count = self._data[target].count()
        unique = self._data[target].nunique()

        rows_to_drop = []
        print(self._data[target].value_counts().to_dict())

        for key, value in self._data[target].value_counts().to_dict().items():
            min_values = count / 100
            if value < min_values:
                rows_to_drop.append(key)

        print("rows to drop: {}".format(rows_to_drop))
        self._data = self._data[self._data[target].isin(rows_to_drop) == False]
        print(self._data[target].value_counts().to_dict())

        ''' @@ Specify target and features '''
        target_column = self._data[target]  # Separates Assembly Code from features
        features = self._data.drop(target, axis=1)  # Separates Features from Assembly Code

        ''' @@ Split data into training and test '''
        training_features, test_features, training_labels, test_labels = train_test_split(
            features, target_column, test_size=0.2, random_state=1, shuffle=True, stratify=target_column)

        ''' @@ Get Category columns and Numerical columns '''
        categorical_cols = self.get_categorical_columns(features)  # Names of string columns
        numerical_cols = self.get_numerical_columns(features)  # Names of numerical columns

        ''' @@ Transform training_features & test_features '''
        training_features, test_features = self.transform_cols(
            training_features, test_features, categorical_cols, numerical_cols)

        ''' @@ Categorize Labels '''
        training_labels, test_labels = self.transform_labels(
            training_labels=training_labels, test_labels=test_labels, target=target)

        """ @ Creates Oversampled Dataset from original"""
        smote = SMOTE(k_neighbors=1)  # Synthetic Minority Oversampling Technique
        training_features_smote, training_labels_smote = smote.fit_resample(training_features, training_labels)

        self.print_data(  # Prints number of elements in each DataFrame
            training_features=training_features,
            test_features=test_features,
            training_labels=training_labels,
            test_labels=test_labels,
            training_features_smote=training_features_smote,
            training_labels_smote=training_labels_smote)

        self._processed_data["training_features"] = training_features
        self._processed_data["training_labels"] = training_labels
        self._processed_data["training_features_smote"] = training_features_smote
        self._processed_data["training_labels_smote"] = training_labels_smote
        self._processed_data["test_features"] = test_features
        self._processed_data["test_labels"] = test_labels

    @property
    def processed_data(self):
        return self._processed_data

    @classmethod
    def print_data(cls, training_features: DataFrame, test_features: DataFrame,
                   training_labels: DataFrame, test_labels: DataFrame,
                   training_features_smote: DataFrame, training_labels_smote: DataFrame):
        print("\nTraining Features: {0}"  # Display number of rows
              "\nTest Features: {1}"
              "\nTraining Labels: {2}"
              "\nTest Labels: {3}"
              "\nTraining Features Smote: {4}"
              "\nTraining Labels Smote: {5}".format(
            len(training_features), len(test_features),
            len(training_labels), len(test_labels),
            len(training_features_smote), len(training_labels_smote)))

    @classmethod
    def get_categorical_columns(cls, features):
        return [cname for cname in features.columns  # Select categorical columns
                if features[cname].dtype == "object"  # IF the type is of object (string)
                and features[cname].nunique() < 10]  # And has less than 10 unique features

    @classmethod
    def get_numerical_columns(cls, features):
        return [cname for cname in features.columns  # Select numerical columns
                if features[cname].dtype  # If the type is in the list with int and float
                in ['int64', 'float64']]

    @classmethod
    def transform_cols(cls, training_features: DataFrame, test_features: DataFrame, categorical_cols: list,
                       numerical_cols: list):
        le = LabelEncoder()  # Label Encoder
        qt = QuantileTransformer(n_quantiles=200, random_state=2)  # QuantileTransformer

        ''' @ Transform 'object' columns to int columns (categorical columns) '''
        training_features.loc[:, categorical_cols] = le.fit_transform(
            training_features.loc[:, categorical_cols].values.ravel())
        test_features.loc[:, categorical_cols] = le.transform(test_features.loc[:, categorical_cols].values.ravel())

        ''' @ Transform numerical columns to values between 0 and 1 '''
        training_features.loc[:, numerical_cols] = qt.fit_transform(training_features.loc[:, numerical_cols])
        test_features.loc[:, numerical_cols] = qt.transform(test_features.loc[:, numerical_cols])

        ''' @ Keep selected columns '''
        my_cols = categorical_cols + numerical_cols
        training_features = training_features[my_cols].copy()
        test_features = test_features[my_cols].copy()

        return training_features, test_features

    @classmethod
    def transform_labels(cls, training_labels, test_labels, target: str):
        enc = LabelEncoder()  # Label Encoder
        training_labels = enc.fit_transform(training_labels)
        test_labels = enc.transform(test_labels)
        # self.mapped_labels = dict(zip(enc.transform(enc.classes_), enc.classes_))
        ''' @@ Numpy Array to DataFrame '''
        training_labels = pd.DataFrame(data=training_labels, columns=[target], index=None)
        test_labels = pd.DataFrame(data=test_labels, columns=[target], index=None)

        return training_labels, test_labels


if __name__ == '__main__':
    names = ["Area", "Structural", "Assembly Code"]
    query_string = ''.join([str("[" + column + '], ') if names.index(column) != len(names) - 1
                            else str("[" + column + "]") for column in names])
    query_final = "SELECT {} FROM [Table]".format(query_string)

    print(query_final)
