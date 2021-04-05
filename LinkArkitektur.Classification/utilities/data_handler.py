"""
     Automates the decitions we make cleaning and selecting data
"""
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self):
        self.self = self
        self.mapped_labels = None

    def preprocessing_data(self, features: DataFrame, target: DataFrame):
        # Split data into training and test
        training_features, test_features, training_labels, test_labels = self.split_data(
            features=features, target=target, target_size=0.2, state=1, should_shuffle=True, stratify=target)

        # Get Category columns and Numerical columns
        categorical_cols = self.get_categorical_columns(features)
        numerical_cols = self.get_numerical_columns(features)

        # Transform columns
        training_features, test_features = self.transform_cols(
            training_features, test_features, categorical_cols, numerical_cols)

        # Categorize labels
        training_labels, test_labels = self.transform_labels(training_labels=training_labels, test_labels=test_labels)

        # Display size of data
        print("\nTraining Features: {0}"
              "\nTest Features: {1}"
              "\nTraining Labels: {2}"
              "\nTest Labels: {3}".format(len(training_features), len(test_features),
                                          len(training_labels), len(test_labels)))

        # Save as csv
        training_features.to_csv(r'..\..\Data\training_features.csv')
        test_features.to_csv(r'..\..\Data\test_features.csv')

        return training_features, test_features, training_labels, test_labels

    def split_data(self,
                   features: DataFrame,
                   target: DataFrame,
                   target_size: float,
                   state: int,
                   should_shuffle: bool,
                   stratify) -> DataFrame:

        print("\n@SPLIT TRAINING AND TESTING DATA SETS")
        training_features, test_features, training_labels, test_labels = train_test_split(
            features, target, test_size=target_size, random_state=state, shuffle=should_shuffle, stratify=stratify)

        print("Training Features: {0}"
              "\nTest Features: {1}"
              "\nTraining Labels: {2}"
              "\nTest Labels: {3}".format(len(training_features), len(test_features),
                                          len(training_labels), len(test_labels)))
        return training_features, test_features, training_labels, test_labels

    def get_categorical_columns(self, features):
        # Select categorical columns
        categorical_cols = [cname for cname in features.columns
                            if features[cname].dtype == "object"
                            and features[cname].nunique() < 10]
        print("\nCatagorical Columns:", categorical_cols)
        return categorical_cols

    def get_numerical_columns(self, features):
        # Select numerical columns
        numerical_cols = [cname for cname in features.columns
                          if features[cname].dtype
                          in ['int64', 'float64']]
        print("\nNumerical Columns:", numerical_cols)
        return numerical_cols

    def transform_cols(self, training_features: DataFrame,test_features:DataFrame, categorical_cols: list(), numerical_cols: list()):
        # Label Encoder
        le = LabelEncoder()

        # QuantileTransformer
        qt = QuantileTransformer(n_quantiles=200, random_state=2)

        # Transform 'object' columns to int columns (categorical columns)
        training_features.loc[:, categorical_cols] = le.fit_transform(
            training_features.loc[:, categorical_cols].values.ravel())
        test_features.loc[:, categorical_cols] = le.transform(test_features.loc[:, categorical_cols].values.ravel())

        # Transform numerical columns to values between 0 and 1
        training_features.loc[:, numerical_cols] = qt.fit_transform(training_features.loc[:, numerical_cols])
        test_features.loc[:, numerical_cols] = qt.transform(test_features.loc[:, numerical_cols])

        # Keep selected columns
        my_cols = categorical_cols + numerical_cols
        training_features = training_features[my_cols].copy()
        test_features = test_features[my_cols].copy()

        return training_features, test_features

    def transform_labels(self, training_labels, test_labels):
        # Label Encoder
        enc = LabelEncoder()
        training_labels = enc.fit_transform(training_labels)
        test_labels = enc.transform(test_labels)
        self.mapped_labels = dict(zip(enc.transform(enc.classes_), enc.classes_))

        return training_labels, test_labels


