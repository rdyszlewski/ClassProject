import pandas as pd
import numpy as np
from sklearn import datasets


class Dataset:

    def __init__(self, data_path):
        self._df = pd.read_csv(data_path)
        # self._df = datasets.load_iris()

        self._df.apply(pd.to_numeric, errors='ignore')
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None
        self._X_train_std = None
        self._X_test_std = None

        self._use_standarized = True

    def init(self, target_column, parameters_column, string_columns = []):

        for index in string_columns:
            column = self._df.columns.values[index]
            class_mapping = {label: idx for idx, label in enumerate(np.unique(self._df[column]))}
            self._df[column] = self._df[column].map(class_mapping)
        self._y = self._df.iloc[0:, target_column].values
        self._X = self._df.iloc[0:, parameters_column].values.astype(float)

        #iris
        # self._y = self._df.target
        # self._X = self._df.data[:, :2]
    #
    def sum_missing_data(self):
        print(self._df.isnull().sum())

    def drop_missing_data(self):
        # TODO usuwanie brakujących rekordów nie działa
        self._df.replace(["NaN", 'NaT'], np.nan, inplace=True)
        self._df.dropna()

    def divide_data(self, test_size=0.3, random_state=1):
        from distutils.version import LooseVersion as Version
        from sklearn import __version__ as sklearn_version
        if Version(sklearn_version) < '0.18':
            from sklearn.grid_search import train_test_split
        else:
            from sklearn.model_selection import train_test_split

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X, self._y, test_size=test_size, random_state=random_state)
        self.standarize()

    def standarize(self):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        sc.fit(self._X)
        self._X_train_std = sc.transform(self._X_train)
        self._X_test_std = sc.transform(self._X_test)

    def set_use_standarized(self, use):
        self._use_standarized = use

    def get_X_train(self):
        if self._use_standarized:
            return self._X_train_std
        else:
            return self._X_train

    def get_X_test(self):
        if self._use_standarized:
            return self._X_test_std
        else:
            return self._X_test

    def get_y_train(self):
        return self._y_train

    def get_y_test(self):
        return self._y_test

    def get_X(self):
        return self._X

    def get_y(self):
        return self._y

    def get_X_combined(self):
        # return np.vstack((self._X_train, self._X_test))
        return np.vstack((self.get_X_train(), self.get_X_test()))

    def get_y_combined(self):
        return np.hstack((self._y_train, self._y_test))

