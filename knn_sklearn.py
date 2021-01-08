import h5py
import sys
import numpy as np
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
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
"""
print(predictor.shape) #Les capteurs
print(target.shape) #classe du signe
print(user.shape) #L'utilisateur
print(predictor[10,1200])
print(type(predictor))

"""
features=predictor.T#list(zip(predictor[1],predictor[2],predictor[]))
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.2)

knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#predicted = knn.predict()
# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
#print(iris.target_names[knn.predict([[3, 5, 4, 2]])])
