import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import plot_confusion_matrix


class PlotHandler:
    def __init__(self):
        print("Plot Handler Created")

    def plot_correlation(self, data: DataFrame):
        # Plot correlation's matrix to explore dependency between features
        # Init figure size
        rcParams['figure.figsize'] = 30, 20
        fig = plt.figure()
        sns.heatmap(data.corr(), annot=True, fmt=".2f")
        plt.show()

    def plt_confusion_matrix(self, test_features, test_labels, model):
        plot_confusion_matrix(model, test_features, test_labels)
        plt.show()