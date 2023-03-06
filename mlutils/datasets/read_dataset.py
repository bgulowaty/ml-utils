import pandas as pd
from box import Box
from loguru import logger


def read_datafile(path):
    logger.debug("Reading path={}", path)
    data = pd.read_csv(path)
    x = data.drop('TARGET', axis=1).values
    y = data['TARGET'].values

    return {
        "x": x,
        "y": y
    }


def right_replace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def read_dataset(path):
    if 'test' in path:
        test_path = path
        train_path = right_replace(path, 'test', 'train', 1)
    else:
        train_path = path
        test_path = right_replace(path, 'train', 'test', 1)

    logger.info("Reading train={} and test={}", train_path, test_path)

    return Box({
        'train': read_datafile(train_path),
        'test': read_datafile(test_path),
        'name': train_path.split("/")[-1].replace("-train", '')
    })
