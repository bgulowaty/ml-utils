import pandas as pd
from box import Box


def read_datafile(path):
    data = pd.read_csv(path)
    x = data.drop('TARGET', axis=1).values
    y = data['TARGET'].values

    return {
        "x": x,
        "y": y
    }


def read_dataset(path):
    if 'test' in path:
        test_path = path
        train_path = path.replace('test', 'train')
    else:
        train_path = path
        test_path = path.replace('train', 'test')

    return Box({
        'train': read_dataset(train_path),
        'test': read_dataset(test_path),
        'name': train_path.split("/")[-1].replace("-train", '')
    })
