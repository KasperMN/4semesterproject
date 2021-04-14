from pandas import DataFrame

import MachineLearning

class TheHandler:
    def __init__(self, data: DataFrame, target: str):
        p = MachineLearning.PreProcessing(data)
        p.returns_processed_test_and_training_data(target)
        mh = MachineLearning.ModelHandler(p.get_data())
        ah = MachineLearning.AccuracyHandler(p.test_features, p.test_labels)

        original_models = mh.create_models()
        smote_models = mh.create_smote_models()

        l = [mh.return_model_cv(model) for model in original_models]
        [ah.add_score(str(type(model))) for model in mh.train_models(l)]

        l2 = [mh.return_model_cv(model) for model in smote_models]
        [ah.add_score(str(type(model))) for model in mh.train_smote_models(l2)]

    def method(self):
