import os

import joblib
import numpy as np
from tqdm import tqdm
from catboost import CatBoostRegressor
from sklearn.metrics import auc
from sklearn.model_selection import KFold

import config as c


def get_combine_predictions(x, model_sale_flg, model_sale_amount):

    target = model_sale_flg.predict(x) * model_sale_amount.predict(x)
    target = target > c.CALL_COST * 1.5
    return target.astype('int')


class Model:
    """Docs for """

    def __init__(self, model_name, k_fold_n_splits=10):
        self.catboost_regressor_models = []
        self.model_name = model_name
        self.k_fold_n_splits = k_fold_n_splits

        for i in range(self.k_fold_n_splits):
            self.catboost_regressor_models.append(self.get_catboost_regressor_model())

    @staticmethod
    def get_catboost_regressor_model():
        model = CatBoostRegressor(iterations=100,
                                  learning_rate=3e-2,
                                  l2_leaf_reg=3.0,  # any pos value
                                  depth=5,  # int up to 16
                                  min_data_in_leaf=1,  # 1,2,3,4,5
                                  rsm=1,  # 0.01 .. 1.0
                                  langevin=False,
                                  task_type="GPU",
                                  devices='0:1')
        return model

    def fit(self, X, y) -> float:
        kf = KFold(n_splits=self.k_fold_n_splits)
        score = 0

        for i, (train_index, test_index) in tqdm(enumerate(kf.split(X))):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.catboost_regressor_models[i].fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)

            score += self.catboost_regressor_models[i].score(X_test, y_test)
        score = score / self.k_fold_n_splits
        print('Models score:', score)
        return score

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
