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

""" @ Collect data from CSV files in GitHub Repository """
training_features = pd.read_csv(r'..\..\Data\training_features.csv', sep=',')  # Training features
test_features = pd.read_csv(r'..\..\Data\test_features.csv', sep=',')  # Testing features
training_labels = pd.read_csv(r'..\..\Data\training_labels.csv', sep=',')  # Training labels
test_labels = pd.read_csv(r'..\..\Data\test_labels.csv', sep=',')  # Testing labels
training_features_smote = pd.read_csv(r'..\..\Data\training_features_smote.csv')  # Oversampled data
training_labels_smote = pd.read_csv(r'..\..\Data\training_labels_smote.csv')  # Oversampled data

# Create the model
network = neuralNetwork.DeepNeuralNetwork(
    features=neuralNetwork.generate_features(feature_set=training_features),
    hidden_layers=[50, 40, 30, 20],
    output_number=13,
    activation_func='relu',
    optimizer='Adam',
    training_features=training_features,
    training_labels=training_labels)

# Instantiate network
network.create_network()

# Train the network
network.train_clf(batch_size=512, training_steps=10000)

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

