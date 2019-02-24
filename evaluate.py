import numpy as np
from utils import train_offset, digit
from scipy.optimize import fmin_powell
from prediction import prediction
import logging


class Within_n_rank():
    def __init__(self, offset, y=0, y_pred=0):
        self.offset = offset

    def score(self, y, y_pred):
        return len(y[abs(y-y_pred) <= self.offset]) / len(y)

    def __call__(self, y, y_pred):
        return self.score(y, y_pred)


def y_transform(train_y_pred, y, y_pred, x0, maxiter=5000):
    offsets = fmin_powell(train_offset, x0, (y, train_y_pred), maxiter=maxiter,
                          disp=True)
    y_pred = digit(offsets, y_pred)
    return y_pred


def cross_validation(model_list, x, y, splitter, scorer_list, average=None,
                     y_tranformation=False, **kargs):
    results = {}
    results['train_index'] = []
    results['val_index'] = []
    for model_name, model in model_list.items():
        results[model_name] = {}
        for scorer_name, scorer in scorer_list.items():
            results[model_name][scorer_name] = []
    for train_index, val_index in splitter.split(x, y):
        train_x, train_y = x.iloc[train_index, :], y[train_index]
        valid_x, valid_y = x.iloc[val_index, :], y[val_index]
        results['train_index'].append(train_index)
        results['val_index'].append(val_index)
        for model_name, model in model_list.items():
            if model_name[:10] == "Binary_XGB":
                threshold = int(model_name[-1])-1
                train_y = np.array(train_y > threshold).astype(int)
            model.fit(train_x, train_y)
            train_y_pred = model.predict(train_x)
            y_pred = model.predict(valid_x)
            if model_name[:10] == "Binary_XGB":
                threshold = int(model_name[-1])-1
                valid_y = np.array(valid_y > threshold).astype(int)
            if y_tranformation:
                y_pred = y_transform(train_y_pred, train_y, y_pred, **kargs)
            for scorer_name, scorer in scorer_list.items():
                try:
                    score = scorer(valid_y, y_pred)
                except ValueError:
                    score = scorer(valid_y, y_pred, average=average)
                print(f"{model_name}: {scorer_name}: {score}")
                results[model_name][scorer_name].append(score)
    for model_name, model in model_list.items():
        for scorer_name, scorer in scorer_list.items():
            results[model_name][f"{scorer_name}_mean"] = np.mean(results[
                    model_name][scorer_name])
    return results


def evaluate(model, data_loader, scorer_list, ts_writer, cuda,
             average='macro'):
    model.eval()
    for scorer_name, scorer in scorer_list.items():
        valid_y, y_pred = prediction(model, data_loader, cuda, label=True)
        try:
            score = scorer(valid_y, y_pred)
        except ValueError:
            score = scorer(valid_y, y_pred, average=average)
        try:
            # change the sign according to your usage
            if score > model.config[scorer_name]:
                model.config[scorer_name] = score
        except KeyError:
            model.config[scorer_name] = score
        ts_writer["tensorboard_writer"].add_scalar(scorer_name, score,
                                                   model.config["global_step"])
        logging.info(f"{scorer_name}: {score}")
    model.train()
