import numpy as np
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit

class Perceptron(object):

    def __init__(self,X_train, y_train, X_test, y_test):
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test

    def start(self, eta=0.01, n_iter=10, random_seed=1):
        X_train = self.x_train
        y_train = self.y_train
        X_test = self.x_test
        y_test = self.y_test

        #uczenie perceptronu
        ppn = SkPerceptron(max_iter=n_iter, eta0=eta, random_state=random_seed)
        ppn.fit(X_train, y_train)


        #wyliczenie dokładności perceptrony
        y_pred = ppn.predict(X_test)
        print('Dokładność: %.2f' % accuracy_score(y_test, y_pred))



