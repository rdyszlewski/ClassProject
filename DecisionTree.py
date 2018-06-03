from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

class DecisionTree(object):

    def __init__(self,X_train, y_train, X_test, y_test):
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test

    def start(self, criterion='entropy', max_depth=3, random_state=0):
        X_train = self.x_train
        y_train = self.y_train
        X_test = self.x_test
        y_test = self.y_test

        # uczenie sieci neuronowej
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)

        # TODO narysować wykres

        y_pred = tree.predict(X_test)
        print(accuracy_score(y_test, y_pred))