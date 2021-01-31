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
from time import time

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
gpc = GaussianProcessClassifier(kernel=kernel,multi_class='one_vs_one',n_restarts_optimizer=10,max_iter_predict=100,n_jobs=5)
t0 = time()
gpc.fit(X_train, t_train)
t1 = time()
time_train_without_validation = t1-t0
t0 = time()
t_pred_without_validation = gpc.predict(X_test)
t1 = time()
time_test_without_validation = t1-t0
acc_without_validation = metrics.accuracy_score(t_test, t_pred_without_validation)
report_without_validation = classification_report(t_test,t_pred_without_validation)


#--------------------CROSS VALIDATION-------------------------------

tuned_parameters = [{'kernel': [1.0 * RBF(1.0), 2.0 * RBF(1.0), 5.0 * RBF(1.0), 10.0 * RBF(1.0)], 'multi_class': ['one_vs_one'], 'n_restarts_optimizer':[10], 'n_jobs':[5]}]

scores = ['precision', 'recall']

t0 = time()

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    gpc = GridSearchCV(
        GaussianProcessClassifier(), tuned_parameters, scoring='%s_macro' % score
    )
    gpc.fit(X_train, t_train)

    print("Best parameters set found on development set:")
    print()
    print(gpc.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = gpc.cv_results_['mean_test_score']
    stds = gpc.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gpc.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    t_true, t_pred = t_test, gpc.predict(X_test)
    print(metrics.classification_report(t_true, t_pred))
    print()

t1 = time()
time_cross_val = t1 - t0
print(gpc.best_params_)

gpc = GaussianProcessClassifier(multi_class=gpc.best_params_['multi_class'],kernel=gpc.best_params_['kernel'],n_restarts_optimizer=gpc.best_params_['n_restarts_optimizer'],n_jobs=5)
#gpc = GaussianProcessClassifier(kernel=(1.41*1.41)*RBF(1.0), multi_class='one_vs_one', n_jobs=5,n_restarts_optimizer=10)
t0 = time()
gpc.fit(X_train, t_train)
t1 = time()
time_train_with_cross_validation = t1 - t0

t0 = time()
t_pred_cross_val = gpc.predict(X_test)
t1 = time()
time_test_with_cross_validation = t1 - t0


print("Accuracy test without validation stage:",acc_without_validation)
print(report_without_validation)
print("Time train exec : ", time_train_without_validation)
print("Time test exec : ", time_test_without_validation)
print("Accuracy test after cross-validation:",metrics.accuracy_score(t_test, t_pred_cross_val))
print(classification_report(t_test,t_pred_cross_val))
print("Time train exec : ", time_train_with_cross_validation)
print("Time test exec : ", time_test_with_cross_validation)
print("Time cross val exec : ", time_cross_val)