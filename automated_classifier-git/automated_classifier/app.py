import os
from automated_classifier import data as dt
from automated_classifier import machinelearning as ml
import joblib


def find_best_classifier(link: str, columns: list, target: str, table_name: str):
    # Extract Data From External Api
    data_collector = dt.DataCollector(chosen_columns=columns)  # Create instance
    data_collector.api_data = link  # Get data from link
    chosen_data = data_collector.chosen_data  # Collect chosen columns from api data
    if data_collector.error_messages:  # Check for error messages
        return {'Status Code: 400': data_collector.error_messages}  # return messages and status code

    # Insert Data Into Database
    db_con = dt.Connection(table_name=table_name, data_to_insert=chosen_data, columns_to_select=columns)  # create database
    db_con.create_database()  # Create the connection and database
    db_con.insert_data()  # Insert the data
    test_query = "SELECT * FROM Walls LIMIT 300"
    unprocessed_data = db_con.get_data(sql=test_query, connection=db_con.connection)

    # then preprocess the data from database
    pre_processor = dt.PreProcessing(data=unprocessed_data)
    pre_processor.create_processed_data(target=target)
    preprocessed_data = pre_processor.processed_data

    # then create the models
    model_handler = ml.ModelHandler(data=preprocessed_data)
    model_handler.create_models()

    model_handler.fit_models()
    fitted_models = model_handler.fitted_models

    accuracy_handler = ml.AccuracyHandler(preprocessed_data["test_features"], preprocessed_data["test_labels"])
    for key, model in fitted_models.items():
        accuracy_handler.add_score(key, model)
    accuracy_handler.display_scores()

    model = fitted_models[accuracy_handler.best_model_name]
    filename = "program/trained_model.sav"
    if os.path.exists(r"program/trained_model.sav"):
        os.remove(r"program/trained_model.sav")
    joblib.dump(model, filename)  # Save the best classifier in folder as file to load

    best_model_score = accuracy_handler.best_model_score  # Return the statistics of the best model and status code
    return {"Model Name": accuracy_handler.best_model_name, 'Model Score': best_model_score}


def predict():
    # Load the model
    #filename = "program/trained_model.sav"
    #loaded_model = joblib.load(filename=filename)
    #result = loaded_model.score(data)
    # Make predictions
    # Return the predictions as file or strings
    return None
