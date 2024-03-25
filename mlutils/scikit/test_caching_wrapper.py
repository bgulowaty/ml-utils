import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from .caching_wrapper import CachingClassifier, joblib_caching_wrapper
import numpy as np
from copy import deepcopy
from sklearn.base import clone
from joblib.memory import Memory



def test_class_caching_wrapper_rf():
    x, y = load_iris(return_X_y=True)
    rf = RandomForestClassifier()
    rf.fit(x, y)
    cached_rf = CachingClassifier(rf)

    assert np.array_equal(rf.predict(x), cached_rf.predict(x))
    assert np.array_equal(rf.predict_proba(x), cached_rf.predict_proba(x))


def test_class_joblib_caching_wrapper_rf():
    memory = Memory("/tmp/joblib-cache")
    x, y = load_iris(return_X_y=True)
    rf = RandomForestClassifier()
    rf.fit(x, y)

    cached_rf = joblib_caching_wrapper(rf)

    assert np.array_equal(rf.predict(x), cached_rf.predict(x))
    assert np.array_equal(rf.predict_proba(x), cached_rf.predict_proba(x))

def test_class_caching_wrapper_dt():
    x, y = load_iris(return_X_y=True)
    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    cached_dt = CachingClassifier(dt)

    assert np.array_equal(dt.predict(x, check_input=True), cached_dt.predict(x, check_input=True))
    assert np.array_equal(dt.predict_proba(x, check_input=True), cached_dt.predict_proba(x, check_input=True))


def test_joblib_caching_wrapper_dt():
    x, y = load_iris(return_X_y=True)
    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    cached_dt = joblib_caching_wrapper(dt)

    assert np.array_equal(dt.predict(x, check_input=True), cached_dt.predict(x, check_input=True))
    assert np.array_equal(dt.predict_proba(x, check_input=True), cached_dt.predict_proba(x, check_input=True))


def test_class_caching_wrapper_rf_elements():
    x, y = load_iris(return_X_y=True)
    rf = RandomForestClassifier()
    rf.fit(x, y)

    cached_rf = deepcopy(rf)
    cached_rf.estimators_ = [CachingClassifier(it) for it in cached_rf.estimators_]

    assert np.array_equal(rf.predict(x), cached_rf.predict(x))
    assert np.array_equal(rf.predict_proba(x), cached_rf.predict_proba(x))

def test_joblib_caching_wrapper_rf_elements():
    x, y = load_iris(return_X_y=True)
    rf = RandomForestClassifier()
    rf.fit(x, y)

    cached_rf = deepcopy(rf)
    cached_rf.estimators_ = [joblib_caching_wrapper(it) for it in cached_rf.estimators_]

    assert np.array_equal(rf.predict(x), cached_rf.predict(x))
    assert np.array_equal(rf.predict_proba(x), cached_rf.predict_proba(x))

def test_class_caching_wrapper_dt_can_be_cloned():
    x, y = load_iris(return_X_y=True)
    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    cached_dt = clone(CachingClassifier(dt))
    cached_dt.fit(x,y)

    assert np.array_equal(dt.predict(x, check_input=True), cached_dt.predict(x, check_input=True))
    assert np.array_equal(dt.predict_proba(x, check_input=True), cached_dt.predict_proba(x, check_input=True))

def test_class_caching_wrapper_should_be_fitted():
    x, y = load_iris(return_X_y=True)
    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    cached_dt = CachingClassifier(dt)

    check_is_fitted(cached_dt)

def test_joblib_caching_wrapper_dt_can_be_cloned():
    x, y = load_iris(return_X_y=True)
    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    cached_dt = clone(joblib_caching_wrapper(dt))
    cached_dt.fit(x,y)

    assert np.array_equal(dt.predict(x, check_input=True), cached_dt.predict(x, check_input=True))
    assert np.array_equal(dt.predict_proba(x, check_input=True), cached_dt.predict_proba(x, check_input=True))

def test_class_caching_wrapper_should_be_fitted():
    x, y = load_iris(return_X_y=True)
    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    cached_dt = joblib_caching_wrapper(dt)

    check_is_fitted(cached_dt)

def test_joblib_joblib_wrapper_multiple_clfs():
    x, y = load_iris(return_X_y=True)
    dt = DecisionTreeClassifier()
    bayes = GaussianNB()
    dt.fit(x, y)
    bayes.fit(x ,y)
    cached_dt = joblib_caching_wrapper(dt)
    cached_bayes = joblib_caching_wrapper(bayes)

    assert np.array_equal(dt.predict(x), cached_dt.predict(x,))
    assert np.array_equal(bayes.predict_proba(x), cached_bayes.predict_proba(x))

