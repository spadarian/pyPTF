"""Base functions to load datasets."""
import os

from pandas import read_csv

data_dir = os.path.join(os.path.dirname(__file__), 'data')


def _load_dataset(dataset_name):
    dataset_path = os.path.join(data_dir, dataset_name + '.csv')
    return read_csv(dataset_path)
