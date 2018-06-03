from Dataset import Dataset
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class Classifiers:


    def perceptron(self, max_iter=10, eta=0.01, random_seed=1):
        return Perceptron(max_iter=max_iter, eta0=eta, random_state=random_seed)

    def neural_network(self, max_iter=10, alpha=0.0001, hidden_layers=(10,2), random_state=1, solver="lbgfs"):
        return MLPClassifier(solver=solver, alpha=alpha, max_iter=max_iter, hidden_layer_sizes=hidden_layers, random_state=random_state )

    def knn(self, neighbors = 2, p =2, metric='minkowski'):
        return KNeighborsClassifier(n_neighbors=neighbors, p=p, metric=metric)

    def decision_tree(self, criterion='entropy', max_depth=3, random_state=0):
        return DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)

    def logistic_regression(self, C=1000.0, random_state=0, penalty='l1'):
        return LogisticRegression(C=C, random_state=random_state, penalty=penalty)

    def svm(self, kernel='rbf', random_state=0, gamma=0.10, C=10.0):
        return SVC(kernel=kernel, random_state=random_state, gamma=gamma, C=C)
