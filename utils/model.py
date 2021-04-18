import os

import joblib
import numpy as np
from tqdm import tqdm
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import auc
from sklearn.model_selection import KFold, train_test_split

import config as c


def get_combine_predictions(x, model_sale_flg, model_sale_amount, model_calls_amount):
    sales_amount = model_sale_amount.predict(x)
    x = np.concatenate((x, sales_amount.reshape(-1, 1)), axis=1)

    target = model_sale_flg.predict(x) * sales_amount
    target = target > c.CALL_COST * model_calls_amount.predict(x)
    return target.astype('int')


class Model:
    """Docs for """

    def __init__(self, model_name, k_fold_n_splits=10):
        self.catboost_regressor_models = []
        self.model_name = model_name
        self.k_fold_n_splits = k_fold_n_splits

        for i in range(self.k_fold_n_splits):
            self.catboost_regressor_models.append(self.get_catboost_model())

    @staticmethod
    def get_catboost_model():
        model = CatBoostRegressor(iterations=400,
                                  depth=5, )
        return model

    def fit(self, X, y):
        for i in range(self.k_fold_n_splits):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
            self.catboost_regressor_models[i].fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
        return 0

    def predict(self, X):
        predict = []
        for i in tqdm(range(self.k_fold_n_splits)):
            predict.append(self.catboost_regressor_models[i].predict(X))
        return np.mean(predict, axis=0)

    def save(self):
        joblib.dump(self, os.path.join(c.MODEL_DIR, self.model_name))

    @staticmethod
    def load(model_name):
        return joblib.load(os.path.join(c.MODEL_DIR, model_name))


class ModelClassifier(Model):
    @staticmethod
    def get_catboost_model():
        model = CatBoostClassifier(
            iterations=400,
            eval_metric='AUC',
            depth=5,
        )
        return model


    def fit(self, X, y):
        for i in range(self.k_fold_n_splits):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i, stratify=y)
            self.catboost_regressor_models[i].fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
        return 0

    def predict(self, X):
        predict = []
        for i in tqdm(range(self.k_fold_n_splits)):
            predict.append(self.catboost_regressor_models[i].predict_proba(X)[:, 1])

        proba = np.mean(predict, axis=0)
        return proba
