"""
    Data_handler
    Responsible for handling data via pandas
"""
import pandas as pd
from pandas import DataFrame

import time


class CsvHandler:
    def __init__(self):
        print("@Handler Created")

    def get_csv_data(self, file_route: str, file_separator: str, target_column: str, tabels_to_drop: list()) -> DataFrame:
        print("\n@Collecting data...")
        #time.sleep(2)
        dataframe = pd.read_csv(file_route, sep=file_separator)

        # Check if not null
        if dataframe is not None:
            print("\n@Data collected")
            #time.sleep(2)

            # Specify target and features
            target = dataframe[target_column]
            features = dataframe.drop(tabels_to_drop, axis=1)
            print("\n@Target name:", target_column)

            # Show features names
            print("\n@FEATURES LIST")
            for col in features.columns:
                print("Feature:", col)
                #time.sleep(0.4)
        else:
            print("Error something went wrong")
        return dataframe, target, features
