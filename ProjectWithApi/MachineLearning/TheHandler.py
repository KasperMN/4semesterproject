from pandas import DataFrame

import MachineLearning

class TheHandler:
    def __init__(self, data: DataFrame, target: str):
        self.p = MachineLearning.PreProcessing(data)
        self.p.returns_processed_test_and_training_data(target)
        self.mh = MachineLearning.ModelHandler(self.p.get_data())
        self.ah = MachineLearning.AccuracyHandler(self.p.test_features, self.p.test_labels)

        original_models = self.mh.create_models()
        smote_models = self.mh.create_smote_models()

        l = [self.mh.return_model_cv(model) for model in original_models]
        [self.ah.add_score(str(type(model)), model) for model in self.mh.train_models(l)]

        l2 = [self.mh.return_model_cv(model) for model in smote_models]
        [self.ah.add_score(str(type(model)), model) for model in self.mh.train_smote_models(l2)]


    def method(self):
        return self.ah.get_score()
