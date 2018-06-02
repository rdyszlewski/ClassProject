import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import  Perceptron
from LogisticRegression import LogisticRegression
from DecisionTree import DecisionTree
from KNN import  KNN
from sklearn import datasets
from sklearn.metrics import accuracy_score

df = pd.read_csv("D://liver.csv")
df.apply(pd.to_numeric, errors='ignore')
y = df.iloc[0:, 10].values # klasy
X = df.iloc[0:, [7]].values.astype(float) # parametry
# df = datasets.load_iris()
# y = df.target
# X = df.data[:,[2,3]]
print(np.unique(y)) # wyświetlenie klas

#podział danych na zestawy
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
else:
    from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

# standaryzacja cech
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

perceptron = Perceptron(X_train_std, y_train, X_test_std, y_test)
perceptron.start(eta= 0.01, n_iter=100, random_seed=1)

regression = LogisticRegression(X_train_std, y_train, X_test_std, y_test)
regression.start()

tree = DecisionTree(X_train_std, y_train, X_test_std, y_test)
tree.start(criterion='entropy', max_depth=8, random_state=1)

knn = KNN(X_train_std, y_train, X_test_std, y_test)
knn.start(neighbors=20, p=2, metric='minkowski')

# ppn = Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.show()

# sieci neuronow