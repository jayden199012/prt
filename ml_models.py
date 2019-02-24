from sklearn import linear_model
from xgboost import XGBClassifier
import numpy as np


class TwoStepRegression():
    def __init__(self, xgb_param, num_cls=8):
        self.xgb_param = xgb_param
        self.xgb_model = XGBClassifier(**xgb_param)
        self.lr = linear_model.LinearRegression()
        self.num_cls = num_cls

    def level_one_predict(self, x):
        # predict clas probabilites
        extra_features = self.xgb_model.predict_proba(x)

        # more models can be added and essembled below:

        # add class probabilities to new x
        x_extra = x.copy()
        for k in range(self.num_cls):
            x_extra[f"pr_{k}"] = extra_features.T[k]
        return x_extra

    def fit(self, x, y):
        self.xgb_model.fit(x, y)
        x_extra = self.level_one_predict(x)
        self.lr.fit(x_extra, y)

    def predict(self, x):
        x_extra = self.level_one_predict(x)
        y_pred = self.lr.predict(x_extra)
        return y_pred


class OrdinalXGBAll():
    def __init__(self, param_list, individual=True, num_cls=7):
        self.param_list = param_list
        self.models = {}
        self.y_pred = []
        if individual:
            for index, param in enumerate(param_list):
                self.models[f"{index+1}"] = XGBClassifier(**param)
        else:
            for index in range(num_cls):
                self.models[f"{index+1}"] = XGBClassifier(**param_list)

    def fit(self, x, y):
        for index, model in enumerate(self.models.values()):
            y_ = y > index
            model.fit(x, y_)

    def predict(self, x):
        self.y_pred = np.ones(x.shape[0])
        for model in self.models.values():
            self.y_pred += model.predict(x)
        return self.y_pred


class OrdinalXGBSeperate():
    def __init__(self, xgb_param, cls=0):
        self.xgb_param = xgb_param
        self.xgb_model = XGBClassifier(**xgb_param)
        self.cls = cls

    def fit(self, x, y):
        y_ = np.array(y > self.cls).astype(int)
        self.xgb_model.fit(x, y_)

    def predict(self, x):
        y_pred = self.xgb_model.predict(x)
        return y_pred

    def __call__(self, x):
        self.predict(x)
