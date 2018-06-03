
import numpy as np
import matplotlib.pyplot as plt
from Dataset import Dataset
from Classifiers import Classifiers

from Teacher import Teacher
import PlotDecisionRegions as pdr
from SBS import SBS

#TODO przekazać jakoś
# target_column = 10
# parameters_columns = [3,7]
# string_columns = [2]
target_column = 10
parameters_columns = [0,1,2,3,4,5,6,7,8]
string_columns = [1]
dataset = Dataset("D://liver.csv")
dataset.init(target_column, parameters_columns,string_columns)
dataset.divide_data(0.3, random_state=30)
dataset.set_use_standarized(True)

# dataset.sum_missing_data()
# dataset.drop_missing_data()
# dataset.sum_missing_data()
classifiers = Classifiers()
teacher = Teacher(dataset)

perceptron = classifiers.perceptron(10, 0.01, 1)
svm = classifiers.svm()
sbs = SBS(svm, k_features=1)
sbs.fit(dataset.get_X_combined(), dataset.get_y_combined())
print(sbs.get_best_subset())
sbs.show_plot()




# perceptron = classifiers.perceptron(10, 0.01, 1)
# print(teacher.fit_and_predict(perceptron))
#
# knn = classifiers.knn(5, 2, 'minkowski')
# print(teacher.fit_and_predict(knn))
#
# network = classifiers.neural_network(10, 0.001, solver='sgd')
# print(teacher.fit_and_predict(network))
#
# regression = classifiers.logistic_regression(C=1000.0, random_state=1)
# print(teacher.fit_and_predict(regression))
#
# svm = classifiers.svm()
# print(teacher.fit_and_predict(svm))
#
# pdr.plot_decision_regions(dataset, perceptron)


