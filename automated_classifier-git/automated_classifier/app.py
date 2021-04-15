from automated_classifier.data import DataCollector
from automated_classifier.data import Connection

def find_best_classifier(link: str, columns: list, target: str, table_name: str):
    # Extract Data From External Api
    data_collector = DataCollector(chosen_columns=columns)  # Create instance
    data_collector.api_data = link  # Get data from link
    chosen_data = data_collector.chosen_data  # Collect chosen columns from api data
    if data_collector.error_messages:  # Check for error messages
        return {'Status Code: 400': data_collector.error_messages}  # return messages and status code

    # Insert Data Into Database
    db_con = Connection(table_name=table_name, data_to_insert=chosen_data, columns_to_select=columns)  # create database
    connection = db_con.connection  # Create the connection and database
    db_con.insert_data()  # Insert the data
    query = db_con.select_query  # Create the Select query
    data = db_con.get_data(sql=query, connection=connection)
    print("Data Collected")
    # then preprocess the data from database
    # then create the models
    # find hyper parameters
    # fit the models
    # cross validate them
    # Save the best classifier in folder as file to load
    # Return the statistics of the best model and status code
    return {'Status Code': 200}


def predict():
    # Load the model
    # Make predictions
    # Return the predictions as file or strings
    return None
