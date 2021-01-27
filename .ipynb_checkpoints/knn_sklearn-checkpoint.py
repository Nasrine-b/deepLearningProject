#https://levelup.gitconnected.com/knn-failure-cases-limitations-and-strategy-to-pick-right-k-45de1b986428
import h5py
import sys
import numpy as np
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
filename = sys.argv[1]

with h5py.File(filename, "r") as hdf:
    # List all groups
    print("Keys: %s" % hdf.keys())
    data = hdf.get('Predictors')
    # Get the predictors
    predictor = np.array(data)

    # Get the target
    data = hdf.get('Target')
    target = np.array(data)

    # Get the target
    data = hdf.get('User')
    user = np.array(data)

target = target[0]
#   Let's combine all features
features=predictor.T
#   Split the data into train and test (80:20)
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.2)

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#   Problem : we don't have the right value of K
#   Then let's split the data into train, validation and test (80:15:15)
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test,y_test, test_size=0.15)

accuracyList = []
KList = range(1,10)
for i in KList:
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_val = knn.predict(X_val)
    accuracyList.append(metrics.accuracy_score(y_val, y_pred_val))
print(accuracyList)


# Create a figure of size 8x6 inches, 80 dots per inch
fig = plt.figure(figsize=(8, 6), dpi=80)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes


ax.grid(linestyle='--', linewidth=0.5)
ax.scatter(KList,accuracyList, s=100, c="red", alpha=1)
ax.plot(KList, accuracyList,color="blue",linewidth=1.0,linestyle="-")
ax.set_title('Accuracy sur validation test pour $K \in [1;10]$')

ax.set_xlabel(r'$K$')
ax.set_ylabel(r'Accuracy')

plt.show()
