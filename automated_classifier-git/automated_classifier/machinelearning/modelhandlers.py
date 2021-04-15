from automated_classifier.machinelearning import classifiers

class ModelHandler:
    def __init__(self, data: dict):
        self._data = data
        self._org_models = {}
        self._fitted_models = {}

    @property
    def models(self):
        return self._org_models

    @property
    def fitted_models(self):
        return self._fitted_models

    def create_models(self):
        self._org_models["KNeighbors"] = classifiers.KNeighbors()
        self._org_models["GradientBoost"] = classifiers.GradientBoost()
        self._org_models["RandomForest"] = classifiers.RandomForest()

    def fit_models(self):
        for key, classifier in self._org_models.items():
            classifier.find_best_estimator(self._data["training_features"], self._data["training_labels"])
            classifier.find_best_estimator(self._data["training_features_smote"], self._data["training_labels_smote"])
            self._fitted_models[key] = classifier.model
            self._fitted_models[key + "_smote"] = classifier.model

if __name__ == '__main__':
    m = ModelHandler()
    models = m.models

    print(models)


