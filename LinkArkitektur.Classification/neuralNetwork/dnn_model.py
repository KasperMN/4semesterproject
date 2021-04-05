"""
    @@Creates a neural network
"""
import warnings

warnings.filterwarnings("ignore")
import utilities
import neuralNetwork
import pandas as pd

# Settings if printing dataframe
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 20)

# Create handler
csv_handler = utilities.CsvHandler()
data_handler = utilities.DataHandler()
plot_handler = utilities.PlotHandler()

# Collect the modified_data
data, target, features = csv_handler.get_csv_data(
    file_route="C:\\LinkArkitektur\\modified_data.csv",
    file_separator=",",
    target_column="Assembly_Code",
    tabels_to_drop=["Type", "Type_Id", "Assembly_Code"])

# Preprocess data
training_features, test_features, training_labels, test_labels = data_handler.preprocessing_data(
    features=features,
    target=target)

# Create the model
network = neuralNetwork.DeepNeuralNetwork(name='LinkArkitektur',
                                          features=neuralNetwork.generate_features(self=features,
                                                                                   feature_set=training_features),
                                          hidden_layers=[50, 40, 30, 20],
                                          output_number=13,
                                          activation_func='relu',
                                          optimizer='Adam',
                                          training_features=training_features,
                                          training_labels=training_labels)
# Instantiate network
network.create_network()

# Train the network
network.train(batch_size=512, training_steps=10_000)

# Evaluate the network
eval_result = network.classifier.evaluate(input_fn=lambda: network.input_func(features=test_features,
                                                                              labels=test_labels,
                                                                              training=False,
                                                                              batch_size=512))
print('\n@@ ACCURACY TEST @@')
print('\nTest set accuracy: {accuracy:0.3f}'.format(**eval_result))

predictions = network.classifier.predict(
    input_fn=lambda: network.predict_input_func(
        test_features=test_features,
        batch_size=512))

#network.display_predictions(predictions=predictions, mapped_labels=data_handler.mapped_labels)
