from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

def start(df, X, y):
    feat_labels = df.columns[0:]
    forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))
    plt.title('Istotność cech')
    plt.bar(range(X.shape[1]), importances[indices],
                      color = 'lightblue', align='center')
    plt.xticks(range(X.shape[1]),feat_labels[indices], rotation=90 )
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()
