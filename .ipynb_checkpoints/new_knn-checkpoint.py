#https://levelup.gitconnected.com/knn-failure-cases-limitations-and-strategy-to-pick-right-k-45de1b986428
import h5py
import sys
import numpy as np
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib.axes import Axes
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
#filename = sys.argv[1]
dir_dataset_sg = 'data/SG24_dataset.h5'
#dir_dataset_sg = './dataset/SG24_dataset_euler.h5'

# Open H5 file to read
file = h5py.File(dir_dataset_sg,'r')
"""
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
"""
X = file['Predictors']
T = file['Target']
U = file['User']

X = np.array(X).transpose() #features
T = np.array(T).transpose() #target
U = np.array(U).transpose() #user

U = U[:,0]
T = T[:,0]

user = 8
ind_all = np.arange(X.shape[0])
ind_all_u = ind_all[U[ind_all]==user]
ind_all = ind_all[U[ind_all]!=user]
#ind_all : 2280

ind_train, ind_test = train_test_split(ind_all,
                                       shuffle=True,
                                       stratify=T[ind_all],
                                       test_size=600,
                                       random_state=42)
ind_val, ind_test = train_test_split(ind_test,
                                     shuffle=True,
                                     stratify=T[ind_test],
                                     test_size=240,
                                     random_state=41)

#Train = 70% , test = 30% et en fait dans test 50% et validation 50% des 30%
## Set user 8 aside
#size_u = ind_all_u.shape[0]

ind_test = np.concatenate((ind_test, ind_all_u))
X_train = X[ind_train,:]
X_val   = X[ind_val,:]
X_test  = X[ind_test,:]
t_train = T[ind_train]
t_val   = T[ind_val]
t_test  = T[ind_test]
u_train = U[ind_train]
u_val   = U[ind_val]
u_test  = U[ind_test]
"""
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, t_train)
t_pred = knn.predict(X_test)
print("Accuracy:",metrics.accuracy_score(t_test, t_pred))
"""
#   Problem : we don't have the right value of K
#   Then let's split the data into train, validation and test (80:15:15)
accuracyList = []
KList = range(1,10)
for i in KList:
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, t_train)
    t_pred_val = knn.predict(X_val)
    accuracyList.append(metrics.accuracy_score(t_val, t_pred_val))
print(accuracyList)

# Create a figure of size 8x6 inches, 80 dots per inch
fig = plt.figure(figsize=(8, 6), dpi=80)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes


ax.grid(linestyle='--', linewidth=0.5)

ax.scatter(KList,accuracyList, s=10, c="red", alpha=1)
#ax.scatter(X_test[:,3],t_pred,s=50,c=t_test)
ax.plot(KList, accuracyList,color="blue",linewidth=1.0,linestyle="-")
ax.set_title('Accuracy sur validation test pour $K \in [1;10]$')

ax.set_xlabel(r'$K$')
ax.set_ylabel(r'Accuracy')
plt.show()

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, t_train)
t_pred = knn.predict(X_test)
print("Accuracy:",metrics.accuracy_score(t_test, t_pred))
print(classification_report(t_test,t_pred))
