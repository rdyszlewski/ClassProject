
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LogRegression
class LogisticRegression(object):

    def __init__(self,X_train, y_train, X_test, y_test):
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test

    def start(self, C=1000.0, random_state=0):
        X_train = self.x_train
        y_train = self.y_train
        X_test = self.x_test
        y_test = self.y_test

        lr = LogRegression(C=1000.0, random_state=0)
        lr.fit(X_train, y_train)
        # TODO dorobić funkcje rysującą wykres

        # y_pred = lr.predict_proba(X_test[0,:].reshape(1,-1))
        y_pred = lr.predict(X_test)
        print(accuracy_score(y_test, y_pred))
