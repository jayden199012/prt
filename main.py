import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from scipy.optimize import fmin_powell
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import Lasso
from ml_models import TwoStepRegression, OrdinalXGBSeperate, OrdinalXGBAll
from utils import prep_data, quadratic_weighted_kappa, train_offset, digit,\
    feature_importance_plot, simple_score
from evaluate import y_transform, cross_validation, Within_n_rank
import mord
import random
import warnings
import logging
import json
from catboost import CatBoostRegressor, CatBoostClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")
# =============================================================================
# data
# =============================================================================
train, test, target, validation, valid_target,\
 categorical_variables = prep_data(160)
#train, test, target = prep_data(pca=True, pca_scale=True,
#                                remove_low_variance=False)


# =============================================================================
# Pure regression based
# =============================================================================
# lasso
lasso = Lasso(alpha=0.001, max_iter=5000)
lasso.fit(train, target)
train_y_pred = lasso.predict(train)
test_pred = lasso.predict(test)

# drop non important features (optional)
lasso_feature_importance = pd.Series(data=lasso.coef_,
                                     index=train.columns,
                                     name='feature_importance'
                                     ).sort_values(0, ascending=False)

print(str(sum(lasso_feature_importance == 0)) + " Out of " + str(
        sum(lasso_feature_importance != object)) + " features are 0")
feature_importance_plot(lasso_feature_importance)
zero_coef = list(lasso_feature_importance[
                                lasso_feature_importance.values == 0].index)
x_dropped = train.drop(zero_coef, axis=1)
test_x_dropped = test.drop(zero_coef, axis=1)
lasso.fit(x_dropped, target)
train_y_pred = lasso.predict(x_dropped)
test_pred = lasso.predict(test_x_dropped)


# xgb
xgbr = XGBRegressor(objective='count:poisson',
                    n_estimators=500,
                    nthread=4,
                    max_depth=3,
                    min_child_weight=8,
                    subsample=0.7,
                    seed=1,
                    reg_alpha=0.015357894736842106
                    )

xgbr.fit(train, target)
train_y_pred = xgbr.predict(train)
valid_y_pred = xgbr.predict(validation)
test_pred = xgbr.predict(test)
x0 = (0.35, 0.7, 1.05, 1.4)
simple_score(x0, target, train_y_pred, valid_y_pred, valid_target)

from catboost import CatBoost
# CatBoostRegressor
cat_feat = [train.columns.get_loc(x) for x in categorical_variables]
ctbr = CatBoostRegressor(objective='RMSE',
                            n_estimators=4000,
                            max_depth=3,
                            l2_leaf_reg = 4,
                            task_type='GPU',
                            one_hot_max_size=100,
                            random_seed=1,
                            bagging_temperature = 1,
                            learning_rate=0.03,
                            cat_features=cat_feat
                            )

train.columns.get_loc(categorical_variables[0])
ctbr.fit(train, target)
train_y_pred = ctbr.predict(train)
valid_y_pred = ctbr.predict(validation)
test_pred = ctbr.predict(test)
x0 = (0.35, 0.7, 1.05, 1.4)
simple_score(x0, target, train_y_pred, valid_y_pred, valid_target)
# =============================================================================
# 2 step regression
# =============================================================================
# using xgb
xgb_param = {'max_depth': 4,
             'eta': 0.001,
             'silent': 1,
             'min_child_weight': 10,
             'subsample': 0.5,
             'early_stopping_rounds': 100,
             'objective': 'multi:softprob',
             'num_class': 8,
             'n_estimators': 2000,
             'colsample_bytree': 0.3,
             'seed': 0}

tsr = TwoStepRegression(xgb_param, num_cls=8)
tsr.fit(x, y)
train_y_pred = tsr.predict(x)
test_pred = tsr.predict(test_x)
# %%
# =============================================================================
# pure classification
# =============================================================================
xgb_classifer_param = {'max_depth': 4,
                       'eta': 0.001,
                       'silent': 1,
                       'min_child_weight': 10,
                       'subsample': 0.5,
                       'early_stopping_rounds': 100,
                       'objective': 'multi:softmax',
                       'num_class': 8,
                       'n_estimators': 2000,
                       'colsample_bytree': 0.3,
                       'seed': 1}

xgb_classifier = XGBClassifier(**xgb_classifer_param)
xgb_classifier.fit(x, y)
train_pred = xgb_classifier.predict(x)
y_pred = xgb_classifier.predict(test)
# =============================================================================
# 7 XGBoost
# =============================================================================
seed = 1
random.seed(seed)
np.random.seed(seed)
two_off_set = Within_n_rank(2)
num_cls = 7
x0 = (1, 2., 3., 4., 5., 6., 7.)
xgb_binary_param = {'max_depth': 4,
                    'eta': 0.001,
                    'silent': 1,
                    'min_child_weight': 10,
                    'subsample': 0.5,
                    'early_stopping_rounds': 100,
                    'objective': 'binary:logistic',
                    'n_estimators': 2000,
                    'colsample_bytree': 0.3,
                    'seed': 1}

models = {}
for i in range(num_cls):
    model_ = OrdinalXGBSeperate(xgb_binary_param, cls=i)
    models[f"Binary_XGB_{i+1}"] = model_
s = StratifiedShuffleSplit(n_splits=5, test_size=0.2)

seven_xgb = OrdinalXGBAll(xgb_binary_param,individual=False, num_cls=num_cls)
#models["Binary_XGB_1"].fit(x, y)
#models["Binary_XGB_1"].xgb_model.fit(x, y)
#models["Binary_XGB_1"].xgb_model.predict(x)
#scorer_list = {'f1_score': f1_score,
#               'accuracy_score': accuracy_score}
scorer_list = {'quadratic_weighted_kappa': quadratic_weighted_kappa,
               'f1_score': f1_score,
               'two_off_set': two_off_set,
               'accuracy_score': accuracy_score}
model_list = {'seven_xgb': seven_xgb}
results = cross_validation(model_list, x, y, s, scorer_list,
                           y_tranformation=False, average='macro', x0=x0,
                           maxiter=20000)
# =============================================================================
# cross validation for all models
# =============================================================================
# initiate with_n_rank object
one_off_set = Within_n_rank(1)
two_off_set = Within_n_rank(2)

# set seed
seed = 1
random.seed(seed)
np.random.seed(seed)

# initial offset values
# x0 = (1.5, 2.9, 3.1, 4.5, 5.5, 6.1, 7.1)
x0 = (0.35, 0.7, 1.05, 1.4)
s = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
scorer_list = {'quadratic_weighted_kappa': quadratic_weighted_kappa,
               'f1_score': f1_score,
               'one_off_set': one_off_set,
               'two_off_set': two_off_set,
               'accuracy_score': accuracy_score}

model_list = {'ctbr': ctbr}
#model_list = {'lasso': lasso,
#              'xgbr': xgbr,
#              'two_step_regression': tsr}
results_reg = cross_validation(model_list, train, target, s, scorer_list,
                           y_tranformation=y_transform, average='macro', x0=x0,
                           maxiter=20000)


model_list = {'xgb_classifier': xgb_classifier,
              'lad_model_IT': lad_model_IT,
              'lad_model_AT': lad_model_AT}
              
results_cls = cross_validation(model_list, x, y, s, scorer_list,
                           y_tranformation=False, average='macro', x0=x0,
                           maxiter=20000)
# =============================================================================
# Tuning
# =============================================================================
tune_index = results['train_index'][2]
tune_x, tune_y = x.iloc[tune_index, :], y[tune_index]
xgbr.fit(tune_x, tune_y)
train_y_pred = xgbr.predict(tune_x)
test_pred = xgbr.predict(test_x)
offsets = fmin_powell(train_offset, x0, (tune_y, train_y_pred), maxiter=20000,
                      disp=True)
y_pred = np.asarray(digit(offsets, test_pred))
# =============================================================================
# output transformation for regression based models
# =============================================================================
# find the offset
offsets = fmin_powell(train_offset, x0, (target, train_y_pred), maxiter=20000,
                      disp=True)


# in case you need one more time of minimizationg
offsets = fmin_powell(train_offset, offsets, (y, train_y_pred), maxiter=20000,
                      disp=True)

# evaluate
train_y_pred = digit(offsets, train_y_pred)
quadratic_weighted_kappa(target, train_y_pred)

valid_y_pred = digit(offsets, valid_y_pred)
quadratic_weighted_kappa(valid_target, valid_y_pred)

# final predict
y_pred = np.asarray(digit(offsets, test_pred))


# =============================================================================
# submit
# =============================================================================
submission = pd.read_csv('../data/test/sample_submission.csv', index_col=0)
submission['AdoptionSpeed'] = y_pred.astype('int32')
submission.to_csv('submit.csv')

