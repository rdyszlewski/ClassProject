from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


class NeuralNetwork(object):

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

        # uczenie sieci neuronowej
        clf = MLPClassifier(solver='lbfgs', alpha=0.0001, max_iter=10000, hidden_layer_sizes=(10, 2), random_state=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(accuracy_score(y_test, y_pred))
