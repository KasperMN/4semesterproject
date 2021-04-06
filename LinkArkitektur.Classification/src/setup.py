from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class Setup:
    def __init__(self):
        self.data = pd.read_csv("..\\..\\Data\\modified_data.csv", sep=",")  # The complete dataset
        self.target = 'Assembly_Code'  # Target label for classification

        ''' @@ Remove Duplicates From Data '''
        print('Shape of Data: {}'.format(self.data.shape[0]))  # Prints num rows
        self.data.drop_duplicates(inplace=True) # Drops duplicate rows
        print('Shape of Data after dropping duplicates: {}'.format(self.data.shape[0]))  # Prints num rows

        ''' @@ Specify target and features '''
        target = self.data[self.target]  # Separates Assembly Code from features
        features = self.data.drop(self.target, axis=1)  # Separates Features from Assembly Code

        ''' @@ Split data into training and test '''
        training_features, test_features, training_labels, test_labels = train_test_split(
            features, target, test_size=0.2, random_state=1, shuffle=True, stratify=target)

        ''' @@ Get Category columns and Numerical columns '''
        categorical_cols = self.get_categorical_columns(features)
        numerical_cols = self.get_numerical_columns(features)

        ''' @@ Transform Columns '''
        training_features, test_features = self.transform_cols(training_features, test_features, categorical_cols, numerical_cols)

        ''' @@ Categorize Labels '''
        training_labels, test_labels = self.transform_labels(training_labels=training_labels, test_labels=test_labels)

        ''' @@ Save as csv '''
        training_features.to_csv(r'..\..\Data\training_features.csv')
        test_features.to_csv(r'..\..\Data\test_features.csv')

        ''' @@ numpy.ndarry to DataFrame '''
        training_labels = pd.DataFrame(data=training_labels, columns=['Assembly_Code'], index=None)
        test_labels = pd.DataFrame(data=test_labels, columns=['Assembly_Code'], index=None)

        ''' @@ Save as csv '''
        training_labels.to_csv(r'..\..\Data\training_labels.csv', index=False)
        test_labels.to_csv(r'..\..\Data\test_labels.csv', index=False)

        self.print_data(training_features, test_features, training_labels, test_labels)


    def print_data(self, training_features: DataFrame, test_features: DataFrame,
                   training_labels: DataFrame, test_labels: DataFrame):
        # Display size of data
        print("\nTraining Features: {0}"
              "\nTest Features: {1}"
              "\nTraining Labels: {2}"
              "\nTest Labels: {3}".format(len(training_features), len(test_features),
                                          len(training_labels), len(test_labels)))

    @staticmethod
    def get_categorical_columns(features):
        # Select categorical columns
        return [cname for cname in features.columns
                if features[cname].dtype == "object"
                and features[cname].nunique() < 10]

    @staticmethod
    def get_numerical_columns(features):
        # Select numerical columns
        return [cname for cname in features.columns
                if features[cname].dtype
                in ['int64', 'float64']]

    @staticmethod
    def transform_cols(training_features: DataFrame, test_features: DataFrame, categorical_cols: list(),
                       numerical_cols: list()):
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


setup = Setup()  # Creates instance to run setup class
