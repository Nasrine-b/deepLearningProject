import h5py
import numpy as np
import random
import csv
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

#%% DATA LOADING

dir_dataset_sg = 'data/SG24_dataset.h5'

# Open H5 file to read
file = h5py.File(dir_dataset_sg,'r')


# Load static gesture data set
X = file['Predictors']
T = file['Target']
U = file['User']

X = np.array(X).transpose()
T = np.array(T).transpose()
U = np.array(U).transpose()
U = U[:,0]

# Dataset statistics
num_users = np.unique(U).shape[0]
for u in np.unique(U):
    print('User %i: %i samples out of total %i (%.1f%%)' % (u, sum(U==u), len(U), sum(U==u)/len(U)*100))

user = 8
ind_all = np.arange(X.shape[0])
ind_all_u = ind_all[U[ind_all]==user]
ind_all = ind_all[U[ind_all]!=user]

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

np.set_printoptions(threshold=np.inf)





ind_all = np.arange(X.shape[0])

ind_train, ind_test = train_test_split(ind_all,
                                       shuffle=True,
                                       stratify=T[ind_all],
                                       test_size=0.3,
                                       random_state=42)

X_train = X[ind_train,:]
X_test  = X[ind_test,:]
t_train = T[ind_train]
t_test  = T[ind_test]
u_train = U[ind_train]
u_test  = U[ind_test]

kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier()
gpc.fit(X_train, t_train)
t_pred_without_validation = gpc.predict(X_test)
acc_without_validation = metrics.accuracy_score(t_test, t_pred_without_validation)
report_without_validation = classification_report(t_test,t_pred_without_validation)
print(acc_without_validation)
print(report_without_validation)