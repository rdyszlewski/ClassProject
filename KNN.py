from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


class KNN(object):

    def __init__(self,X_train, y_train, X_test, y_test):
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test

    def start(self, neighbors = 5, p = 2, metric='minkowski'):
        X_train = self.x_train
        y_train = self.y_train
        X_test = self.x_test
        y_test = self.y_test

        knn =KNeighborsClassifier(n_neighbors=neighbors, p=p, metric=metric)
        knn.fit(X_train, y_train)
        #TODO narysować wykres

        y_pred = knn.predict(X_test)
        print(accuracy_score(y_test, y_pred))