import pandas as pd
from loguru import logger
from box import Box

from mlutils.encoding.refittable_label_encoder import RefitableLabelEncoder
from mlutils.encoding.utils import encode_train_test_to_labels


class Dataset:
    def __init__(self, train, test, name):
        self.train = train
        self.test = test
        self.name = name

    def encode_x_to_labels(self):
        train_x_encoded, test_x_encoded = encode_train_test_to_labels(self.train.x, self.test.x)

        self.train.x = train_x_encoded
        self.test.x = test_x_encoded

        return self

    def encode_y_to_numeric_labels(self):
        le = RefitableLabelEncoder()
        self.train.y = le.fit_transform(self.train.y)
        self.test.y = le.fit_transform(self.test.y)

        return self

    def __str__(self):
        return f"""Dataset(name={self.name}, 
        train_size={len(self.train.x)}\ttrain_labels={set(self.train.y)},
        test_size={len(self.test.x)}\ttest_labels={set(self.test.y)})"""

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_data(cls, train_x, train_y, test_x, test_y, name=None):
        return Dataset(Box({
            "x": train_x,
            "y": train_y
        }),
            Box({
                "x": test_x,
                "y": test_y
            }),
            name
        )

    @classmethod
    def read_dataset(cls, train_path, test_path, name=None):
        train = cls.read_single_dataset(train_path)
        test = cls.read_single_dataset(test_path)

        return Dataset(train, test, name)

    @staticmethod
    def read_single_dataset(path, verbose=False, target_column="Class"):
        data = pd.read_csv(path)

        if verbose:
            logger.debug("Reading path={}", path)

        if target_column not in data.columns:
            last_column = data.columns[-1]

            if verbose:
                logger.debug("Last column = {}", last_column)

            data = data.rename(columns={last_column: "Class"})

        x = data.drop("Class", axis=1).values
        y = data["Class"].values

        return Box({
            "x": x,
            "y": y
        })
