import cachetools
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array


class CachingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, _wrapped, cache_factory=lambda: cachetools.LFUCache(maxsize=100000)):
        self._wrapped = _wrapped
        self._cache = cache_factory()

    def __str__(self):
        return f"Caching wrapper around {self._wrapped}"

    @staticmethod
    def _do_hash(numpy_array):
        return str(numpy_array.tobytes())

    def fit(self, X, y):
        return self._wrapped.fit(X, y)

    def predict(self, X):
        X = check_array(X)
        y_classified = np.zeros(X.shape[0])

        hashes = np.apply_along_axis(CachingClassifier._do_hash, axis=1, arr=X)

        already_cached = np.in1d(hashes, self._cache)

        indices_cached = np.where(already_cached)[0]
        indices_not_cached = np.where(~already_cached)[0]

        if len(indices_not_cached) > 0:
            values_classified = self._wrapped.predict(X[indices_not_cached])
            np.put_along_axis(y_classified, indices_not_cached, values_classified, axis=0)

        if len(indices_cached) > 0:
            values_from_cache = np.array([self._cache[it] for it in hashes[indices_cached]])
            np.put_along_axis(y_classified, indices_cached, values_from_cache, axis=0)

        for x, y in zip(X, y_classified):
            self._cache[CachingClassifier._do_hash(x)] = y

        return y_classified

    def predict_proba(self, X):
        X = check_array(X)
        y_classified = None

        hashes = np.apply_along_axis(CachingClassifier._do_hash, axis=1, arr=X)

        already_cached = np.in1d(hashes, self._cache)

        indices_cached = np.where(already_cached)[0]
        indices_not_cached = np.where(~already_cached)[0]

        if len(indices_not_cached) > 0:
            values_classified = self._wrapped.predict_proba(X[indices_not_cached])
            y_classified = np.zeros((X.shape[0], values_classified.shape[1]))

            np.put_along_axis(y_classified, indices_not_cached.reshape((values_classified.shape[0], 1)),
                              values_classified, axis=0)

        if len(indices_cached) > 0:
            values_from_cache = np.array([self._cache[it] for it in hashes[indices_cached]])
            y_classified = np.zeros((X.shape[0], values_from_cache.shape[1])) if y_classified is None else y_classified

            np.put_along_axis(y_classified, indices_cached.reshape((values_from_cache.shape[0], 1)), values_from_cache,
                              axis=0)

        for x, y in zip(X, y_classified):
            self._cache[CachingClassifier._do_hash(x)] = y

        return y_classified
