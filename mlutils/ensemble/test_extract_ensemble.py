import pandas as pd
import pytest
from PyPruning.RankPruningClassifier import RankPruningClassifier
from box import Box
from mlxtend.classifier import EnsembleVoteClassifier
from loguru import logger
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from mlutils.ensemble.extract_ensemble import extract_classifiers_from_bagging

TRAIN_TEST_TUPLES = [
    ('test-resources/nursery-train-1-s2.csv', 'test-resources/nursery-train-1-s2.csv'),
    ('test-resources/breast-train-0-s1.csv', 'test-resources/breast-test-0-s1.csv'),
    ('test-resources/hepatitis-train-3-s1.csv', 'test-resources/hepatitis-test-3-s1.csv'),
    ('test-resources/abalone-train-2-s2.csv', 'test-resources/abalone-test-2-s2.csv'),
]


def read_dataset(path, clas_column_name="TARGET"):
    data = pd.read_csv(path)
    x = data.drop(clas_column_name, axis=1).values
    y = data[clas_column_name].values

    return Box({
        "x": x,
        "y": y
    })


@pytest.mark.parametrize(
    "train_path,test_path",
    TRAIN_TEST_TUPLES
)
def test_given_test_data_should_have_same_accuracy_as_original_ensemble(train_path, test_path):
    # given
    logger.info("train={}, test={}", train_path, test_path)
    train = read_dataset(train_path)
    test = read_dataset(test_path)

    bagging = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), n_estimators=200, max_samples=0.3)
    bagging.fit(train.x, train.y)

    # when
    extracted_clfs = extract_classifiers_from_bagging(bagging)
    extracted_ensemble = EnsembleVoteClassifier(clfs=extracted_clfs, voting='soft', fit_base_estimators=False)
    extracted_ensemble.fit(train.x, train.y)  # Does nothing

    # then
    extracted_acc = accuracy_score(extracted_ensemble.predict(test.x), test.y)
    bagging_acc = accuracy_score(bagging.predict(test.x), test.y)

    assert extracted_acc == bagging_acc


@pytest.mark.parametrize(
    "train_path,test_path",
    TRAIN_TEST_TUPLES
)
def test_can_be_pruned(train_path, test_path):
    # given
    logger.info("train={}, test={}", train_path, test_path)
    train = read_dataset(train_path)
    le = LabelEncoder()
    train.y = le.fit_transform(train.y)

    bagging = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), n_estimators=200, max_samples=0.3)
    bagging.fit(train.x, train.y)

    # when
    extracted_clfs = extract_classifiers_from_bagging(bagging)

    clf_random_pruned = RankPruningClassifier(n_estimators=20)
    clf_random_pruned.prune(train.x, train.y, extracted_clfs)

    # then
    # passes
