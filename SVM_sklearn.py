import h5py
import numpy as np
import random
import csv
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from sklearn.datasets import make_classification
from sklearn import svm, metrics

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


#SVM with sklearn

#Support Vector Classification with rbf as kernel

#-------------VALIDATION-----------------

acc=[]
ran=range(1,20)

#Validation stage to search for the best value of regularization parameter

for i in ran:
	clf = svm.SVC(C=i, kernel='rbf')
	clf.fit(X_train, t_train)
	t_pred = clf.predict(X_val)
	acc.append(metrics.accuracy_score(t_val, t_pred))
acc=np.array(acc)
#print(acc.max())

'''# Create a figure of size 8x6 inches, 80 dots per inch
fig = plt.figure(figsize=(8, 6), dpi=80)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes


ax.grid(linestyle='--', linewidth=0.5)
ax.scatter(ran,acc, s=100, c="red", alpha=1)
ax.plot(ran, acc,color="blue",linewidth=1.0,linestyle="-")
ax.set_title('Accuracy par rapport à la regularization C.')

ax.set_xlabel(r'Parametre de regularization')
ax.set_ylabel(r'Accuracy')

#plt.show()'''	

clf = svm.SVC(C=ran[np.where(acc==acc.max())[0][0]-1], kernel='rbf')
clf.fit(X_train, t_train)
t_pred_val = clf.predict(X_test)
acc_val = metrics.accuracy_score(t_test, t_pred_val)
report_with_validation = classification_report(t_test,t_pred_val)
'''print("Accuracy without validation stage:",metrics.accuracy_score(t_test, t_pred_without_validation))
print("Accuracy with validation:",metrics.accuracy_score(t_test, t_pred_val))'''

'''clf = svm.SVC(C=1)
scores = cross_val_score(clf,X_train,t_train,cv=5)
#print(scores)
#print("%0.2f accuracy with a standard deviation of %0.2f"%(scores.mean(), scores.std()))'''

#---------------WITHOUT VALIDATION-------------------

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

clf = svm.SVC(C=1, kernel='rbf')
clf.fit(X_train, t_train)

t_pred_without_validation = clf.predict(X_test)
acc_without_validation = metrics.accuracy_score(t_test, t_pred_without_validation)
report_without_validation = classification_report(t_test,t_pred_without_validation)

#--------------------CROSS VALIDATION-------------------------------

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C':[1, 10, 100 ,1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        svm.SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, t_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    t_true, t_pred = t_test, clf.predict(X_test)
    print(metrics.classification_report(t_true, t_pred))
    print()

clf = svm.SVC(C=clf.best_params_['C'],kernel=clf.best_params_['kernel'],gamma=clf.best_params_['gamma'])
clf.fit(X_train, t_train)
t_pred_cross_val = clf.predict(X_test)
t_train_cross_val = clf.predict(X_train)

print("Accuracy test without validation stage:",acc_without_validation)
print(report_without_validation)
print("Accuracy test after validation:",acc_val)
print(report_with_validation)
print("Accuracy test after cross-validation:",metrics.accuracy_score(t_test, t_pred_cross_val))
print(classification_report(t_test,t_pred_cross_val))
print("Accuracy train after cross-validation:",metrics.accuracy_score(t_train, t_train_cross_val))
print(classification_report(t_train, t_train_cross_val))


'''X_test_u = X_test[u_test==user]

t_pred_u = clf.predict(X_test_u)
print(str("Accuracy for user "+str(user)+" only :"),metrics.accuracy_score(t_test[u_test==user], t_pred_u))'''
