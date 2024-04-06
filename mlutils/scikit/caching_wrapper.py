import tempfile
from copy import deepcopy

import cachetools
import numpy as np
import xxhash
import types

from joblib import Memory
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import _is_fitted


class CachingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, _wrapped, cache_factory=lambda: cachetools.LFUCache(maxsize=100000)):
        self._wrapped = _wrapped
        self._cache = cache_factory()
        self._proba_cache = cache_factory()

    def __str__(self):
        return f"Caching wrapper around {self._wrapped.__str__()}"

    def __sklearn_clone__(self):
        clone = CachingClassifier(self._wrapped.__sklearn_clone__())
        clone._cache = self._cache
        clone._proba_cache = self._proba_cache

        return clone

    @staticmethod
    def _do_hash(numpy_array):
        return xxhash.xxh32(np.ascontiguousarray(numpy_array)).hexdigest()
        # return str(numpy_array.tobytes())

    def __len__(self):
        return self._wrapped.__len__()

    def __sklearn_is_fitted__(self, *args, **kwargs):
        return _is_fitted(self._wrapped, *args, **kwargs)

    def fit(self, X, y):
        return self._wrapped.fit(X, y)

    def predict(self, X, *args, **kwargs):
        X = check_array(X)
        y_classified = np.zeros(X.shape[0])

        hashes = np.apply_along_axis(CachingClassifier._do_hash, axis=1, arr=X)

        already_cached = np.in1d(hashes, self._cache)

        indices_cached = np.where(already_cached)[0]
        indices_not_cached = np.where(~already_cached)[0]

        if len(indices_not_cached) > 0:
            values_classified = self._wrapped.predict(X[indices_not_cached], *args, **kwargs)
            np.put_along_axis(y_classified, indices_not_cached, values_classified, axis=0)

        if len(indices_cached) > 0:
            values_from_cache = np.array([self._cache[it] for it in hashes[indices_cached]])
            np.put_along_axis(y_classified, indices_cached, values_from_cache, axis=0)

        for x, y in zip(X, y_classified):
            self._cache[CachingClassifier._do_hash(x)] = y

        return y_classified

    def predict_proba(self, X, *args, **kwargs):
        X = check_array(X)
        y_classified = None

        hashes = np.apply_along_axis(CachingClassifier._do_hash, axis=1, arr=X)

        already_cached = np.in1d(hashes, self._proba_cache)

        indices_cached = np.where(already_cached)[0]
        indices_not_cached = np.where(~already_cached)[0]

        if len(indices_not_cached) > 0:
            values_classified = self._wrapped.predict_proba(X[indices_not_cached], *args, **kwargs)
            y_classified = np.zeros((X.shape[0], values_classified.shape[1]))

            np.put_along_axis(y_classified, indices_not_cached.reshape((values_classified.shape[0], 1)),
                              values_classified, axis=0)

        if len(indices_cached) > 0:
            values_from_cache = np.array([self._proba_cache[it] for it in hashes[indices_cached]])
            y_classified = np.zeros((X.shape[0], values_from_cache.shape[1])) if y_classified is None else y_classified

            np.put_along_axis(y_classified, indices_cached.reshape((values_from_cache.shape[0], 1)), values_from_cache,
                              axis=0)

        for x, y in zip(X, y_classified):
            self._proba_cache[CachingClassifier._do_hash(x)] = y

        return y_classified

    def __getattr__(self, name: str):
        attr = getattr(self._wrapped, name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            return attr(*args, **kwargs)

        return wrapper

def _do_hash(numpy_array):
    return str(numpy_array.tobytes())

def wrap_classifier(classifier, cache_factory = lambda: cachetools.LFUCache(maxsize=100000)):
    original_predict = classifier.predict
    original_predict_proba = classifier.predict_proba
    cache = cache_factory()

    def predict(self, X, *args, **kwargs):
        X = check_array(X)
        y_classified = np.zeros(X.shape[0])

        hashes = np.apply_along_axis(_do_hash, axis=1, arr=X)

        already_cached = np.in1d(hashes, cache)

        indices_cached = np.where(already_cached)[0]
        indices_not_cached = np.where(~already_cached)[0]

        if len(indices_not_cached) > 0:
            values_classified = original_predict(X[indices_not_cached], *args, **kwargs)
            np.put_along_axis(y_classified, indices_not_cached, values_classified, axis=0)

        if len(indices_cached) > 0:
            values_from_cache = np.array([cache[it] for it in hashes[indices_cached]])
            np.put_along_axis(y_classified, indices_cached, values_from_cache, axis=0)

        for x, y in zip(X, y_classified):
            cache[_do_hash(x)] = y

        return y_classified

    def predict_proba(self, X, *args, **kwargs):
        X = check_array(X)
        y_classified = None

        hashes = np.apply_along_axis(_do_hash, axis=1, arr=X)

        already_cached = np.in1d(hashes, cache)

        indices_cached = np.where(already_cached)[0]
        indices_not_cached = np.where(~already_cached)[0]

        if len(indices_not_cached) > 0:
            values_classified = original_predict_proba(X[indices_not_cached], *args, **kwargs)
            y_classified = np.zeros((X.shape[0], values_classified.shape[1]))

            np.put_along_axis(y_classified, indices_not_cached.reshape((values_classified.shape[0], 1)),
                              values_classified, axis=0)

        if len(indices_cached) > 0:
            values_from_cache = np.array([cache[it] for it in hashes[indices_cached]])
            y_classified = np.zeros((X.shape[0], values_from_cache.shape[1])) if y_classified is None else y_classified

            np.put_along_axis(y_classified, indices_cached.reshape((values_from_cache.shape[0], 1)), values_from_cache,
                              axis=0)

        for x, y in zip(X, y_classified):
            cache[_do_hash(x)] = y

        return y_classified

    classifier.predict = types.MethodType(predict, classifier)
    classifier.predict_proba = types.MethodType(predict_proba, classifier)

    return classifier

def sorting_wrapper(predict_func):
    def wrapper(x, *args, **kwargs):
        sort_idx = np.argsort(x)
        desort_idx = np.argsort(sort_idx)

        y_sorted = predict_func(x[sort_idx], *args, **kwargs)

        return y_sorted[sort_idx][desort_idx]

    return wrapper

def joblib_caching_wrapper(clf, verbose=False):
    clf = deepcopy(clf)

    if joblib_caching_wrapper.joblib_memory is None:
        temp_path = tempfile.mkdtemp()
        if verbose:
            print(f"Caching temp path={temp_path}")
        joblib_caching_wrapper.joblib_memory = Memory(temp_path, verbose=1 if verbose else 0)

    clf.predict = joblib_caching_wrapper.joblib_memory.cache(clf.predict)
    clf.predict_proba = joblib_caching_wrapper.joblib_memory.cache(clf.predict_proba)

    return clf

joblib_caching_wrapper.joblib_memory = None
