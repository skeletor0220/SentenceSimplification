import numpy as np
import pandas as pd


class DataLoader(object):
    def __init__(self, path):
        self.path = path

    @staticmethod
    def load_data(self):
        df = pd.read_csv(self.path, names=['src', 'tar'], sep='\t')

        normal = []
        simple = []

        for aline in df.src:
            normal.append(aline)
        for aline in df.tar:
            simple.append(aline)

        return normal, simple

    @staticmethod
    def split_data(self, normal, simple):
        val = np.random.rand(len(normal)) < 0.8
        x_train, x_test = normal[val].copy(deep=True), normal[~val].copy(deep=True)
        y_train, y_test = simple[val].copy(deep=True), simple[~val].copy(deep=True)

        return x_train, x_test, y_train, y_test
