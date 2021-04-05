import pandas as pd
import warnings
import utilities
warnings.filterwarnings("ignore")


class Setup:
    def __init__(self):
        self.data_handler = utilities.DataHandler()
        self.data = pd.read_csv("..\\..\\Data\\modified_data.csv", sep=",")  # The complete dataset
        self.target = 'Assembly_Code'  # Target label for classification

        # Remove Duplicates From Data
        print('Shape of Data: {}'.format(self.data.shape[0]))  # Prints num rows
        self.data.drop_duplicates(inplace=True) # Drops duplicate rows
        print('Shape of Data after dropping duplicates: {}'.format(self.data.shape[0]))  # Prints num rows

        # Specify target and features
        target = self.data[self.target]  # Separates Assembly Code from features
        features = self.data.drop(self.target, axis=1)  # Separates Features from Assembly Code

        # Preprocess data
        self.data_handler.preprocessing_data(features=features, target=target)


setup = Setup()  # Creates instance to run setup class
