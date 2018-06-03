from Dataset import Dataset
from sklearn.metrics import accuracy_score

class Teacher:

    def __init__(self, dataset):
        self._dataset = dataset

    def fit(self, classifier):
        classifier.fit(self._dataset.get_X_train(), self._dataset.get_y_train())

    def predict(self, classifier):
        train_score = classifier.score(self._dataset.get_X_train(), self._dataset.get_y_train())
        test_score = classifier.score(self._dataset.get_X_test(), self._dataset.get_y_test())
        y_pred = classifier.predict(self._dataset.get_X_test())
        return [train_score, test_score, accuracy_score(self._dataset.get_y_test(), y_pred)]

    def fit_and_predict(self, classifier):
        self.fit(classifier)
        return self.predict(classifier)