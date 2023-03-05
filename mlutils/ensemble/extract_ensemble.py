from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class predict_wrapper(object):
    def __init__(self, predict_func, labels):
        self.predict_func = predict_func
        self.labels = labels

    def __call__(self, *args, **kwargs):
        return self.labels[self.predict_func(*args, **kwargs)]




class predict_proba_wrapper(object):
    def __init__(self, predict_func, estimator_classes, ensemble_classes):
        self.predict_func = predict_func
        self.estimator_classes = estimator_classes
        self.ensemble_classes = ensemble_classes

    def __call__(self, *args, **kwargs):
        predictions = self.predict_func(*args, **kwargs)

        rows = predictions.shape[0]
        zeros = np.zeros((rows, len(self.ensemble_classes)))
        zeros[:, self.estimator_classes] += predictions

        return zeros


class ExtractedClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, wrapped, ensemble):
        self._wrapped = wrapped
        self.predict = predict_wrapper(wrapped.predict, ensemble.classes_)
        self.predict_proba = predict_proba_wrapper(wrapped.predict_proba, wrapped.classes_, ensemble.classes_)
        self.classes_ = ensemble.classes_
        self.n_classes_ = ensemble.n_classes_

def raise_not_implemented():
    raise NotImplemented("Predict proba is not supported")


def extract_classifiers_from_bagging(bagging):
    extracted = []
    for classifier in bagging.estimators_:
        cloned_classifier = ExtractedClassifierWrapper(deepcopy(classifier), bagging)
        extracted.append(cloned_classifier)

    return extracted