import sqlite3
import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 20)


class PreProcessing:
    def __init__(self, columns, table_name: str):
        self.conn = sqlite3.connect(r"../DataBase/database.db")
        self.sql = self.create_sql_query(attributes_to_select=columns, table_name=table_name)
        self.data = pd.read_sql(sql=self.sql, con=self.conn)

    def create_sql_query(self, attributes_to_select: list, table_name: str):
        attribute_string = ''.join(
            [str("[" + column + '], ') if attributes_to_select.index(column) != len(attributes_to_select) - 1
             else str("[" + column + "]") for column in attributes_to_select])

        return "Select {0} FROM [{1}]".format(attribute_string, table_name)

    def returns_processed_test_and_training_data(self, target):
        print('Shape of Data: {}'.format(self.data.shape[0]))  # Prints num rows

        ''' @@ Remove Duplicates From Data '''
        self.data.drop_duplicates(inplace=True)  # Drops duplicate rows
        print('Shape of Data after dropping duplicates: {}'.format(self.data.shape[0]))  # Prints num rows

        ''' @@ Specify target and features '''
        target_column = self.data[target]  # Separates Assembly Code from features
        features = self.data.drop(target, axis=1)  # Separates Features from Assembly Code

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
            training_labels=training_labels, test_labels=test_labels)

        ''' @@ Numpy Array to DataFrame '''
        training_labels = pd.DataFrame(data=training_labels, columns=['Assembly_Code'], index=None)
        test_labels = pd.DataFrame(data=test_labels, columns=['Assembly_Code'], index=None)

        """ @ Creates Oversampled Dataset from original"""
        smote = SMOTE(k_neighbors=1)  # Synthetic Minority Oversampling Technique
        training_features_smote, training_labels_smote = smote.fit_resample(training_features, training_labels)

        self.print_data(  # Prints number of elements in each DataFrame
            training_features=training_features,
            test_features=test_features,
            training_labels=training_labels,
            test_labels=test_labels)

    @staticmethod
    def print_data(training_features: DataFrame, test_features: DataFrame,
                   training_labels: DataFrame, test_labels: DataFrame):
        print("\nTraining Features: {0}"  # Display number of rows
              "\nTest Features: {1}"
              "\nTraining Labels: {2}"
              "\nTest Labels: {3}".format(len(training_features), len(test_features),
                                          len(training_labels), len(test_labels)))

    @staticmethod
    def get_categorical_columns(features):
        return [cname for cname in features.columns  # Select categorical columns
                if features[cname].dtype == "object"  # IF the type is of object (string)
                and features[cname].nunique() < 10]  # And has less than 10 unique features

    @staticmethod
    def get_numerical_columns(features):
        return [cname for cname in features.columns  # Select numerical columns
                if features[cname].dtype  # If the type is in the list with int and float
                in ['int64', 'float64']]

    @staticmethod
    def transform_cols(training_features: DataFrame, test_features: DataFrame, categorical_cols: list,
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

    @staticmethod
    def transform_labels(training_labels, test_labels):
        enc = LabelEncoder()  # Label Encoder
        training_labels = enc.fit_transform(training_labels)
        test_labels = enc.transform(test_labels)
        # self.mapped_labels = dict(zip(enc.transform(enc.classes_), enc.classes_))

        return training_labels, test_labels

if __name__ == '__main__':
    names = ["Area", "Structural", "Assembly Code"]
    query_string = ''.join([str("[" + column + '], ') if names.index(column) != len(names) - 1
                            else str("[" + column + "]") for column in names])
    query_final = "SELECT {} FROM [Table]".format(query_string)

    print(query_final)
