from tensorflow.estimator import DNNClassifier
import tensorflow as tf
from pandas import DataFrame

"""
    @ Deep Neural Network Classifier
"""


@tf.autograph.experimental.do_not_convert
class DeepNeuralNetwork:
    def __init__(self,
                 name: str,
                 features,
                 hidden_layers: list,
                 output_number: int,
                 activation_func: str,
                 optimizer: str,
                 training_features: DataFrame,
                 training_labels: DataFrame) -> DNNClassifier:

        self.name = name
        self.features = features
        self.hidden_layers = hidden_layers
        self.output_number = output_number
        self.activation_func = activation_func
        self.optimizer = optimizer
        self.training_features = training_features
        self.training_labels = training_labels
        self.classifier = ''

    def create_network(self):
        network = DNNClassifier(feature_columns=self.features,
                                hidden_units=self.hidden_layers,
                                activation_fn=self.activation_func,
                                optimizer=self.optimizer,
                                n_classes=self.output_number,
                                )
        self.classifier = network
        return self.classifier

    def input_func(self, features: DataFrame,
                   labels: DataFrame,
                   training: bool,
                   batch_size: int) -> tf.data.Dataset:

        data_set = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        if training:
            data_set = data_set.shuffle(10).repeat()
        return data_set.batch(batch_size)

    def predict_input_func(self,
                           test_features: DataFrame,
                           batch_size: int,):
        # Convert the inputs to a Dataset without labels.
        predictions = tf.data.Dataset.from_tensor_slices(dict(test_features)).batch(batch_size)
        return predictions

    def train(self, batch_size: int, training_steps: int) -> DNNClassifier:
        import logging
        logging.getLogger().setLevel(logging.INFO)
        self.classifier = self.classifier.train(
            input_fn=lambda: self.input_func(self.training_features,
                                             self.training_labels,
                                             training=True,
                                             batch_size=batch_size,), steps=training_steps)

        return self.classifier

    def display_predictions(self, predictions, mapped_labels: dict):
        for pre in predictions:
            class_id = pre['class_ids'][0]
            probability = pre['probabilities'][class_id]
            print('Prediction is "{}" ({:.1f}%)'.format(
                mapped_labels.get(class_id), 100 * probability))


def generate_features(self, feature_set):
    my_feature_columns = []
    for key in feature_set.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    return my_feature_columns
