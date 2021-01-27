import h5py
import numpy as np
import random
import csv
from sklearn.model_selection import train_test_split, cross_val_score
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
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

clf = svm.SVC(C=1, kernel='rbf')
clf.fit(X_train, t_train)

t_pred_without_validation = clf.predict(X_test)

acc=[]
ran=range(1,100)

for i in ran:
	clf = svm.SVC(C=i, kernel='rbf')
	clf.fit(X_train, t_train)
	t_pred = clf.predict(X_val)
	acc.append(metrics.accuracy_score(t_val, t_pred))
acc=np.array(acc)
print(acc.max())
# Create a figure of size 8x6 inches, 80 dots per inch
fig = plt.figure(figsize=(8, 6), dpi=80)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes


ax.grid(linestyle='--', linewidth=0.5)
ax.scatter(ran,acc, s=100, c="red", alpha=1)
ax.plot(ran, acc,color="blue",linewidth=1.0,linestyle="-")
ax.set_title('Accuracy par rapport à la regularization C.')

ax.set_xlabel(r'Parametre de regularization')
ax.set_ylabel(r'Accuracy')

#plt.show()	

clf = svm.SVC(C=ran[np.where(acc==acc.max())[0][0]-1], kernel='rbf')
clf.fit(X_train, t_train)
t_pred_val = clf.predict(X_test)
print("Accuracy without validation stage:",metrics.accuracy_score(t_test, t_pred_without_validation))
print("Accuracy with validation:",metrics.accuracy_score(t_test, t_pred_val))

clf = svm.SVC(C=1)
scores = cross_val_score(clf,X_train,t_train,cv=10)
print(scores)

'''X_test_u = X_test[u_test==user]

t_pred_u = clf.predict(X_test_u)
print(str("Accuracy for user "+str(user)+" only :"),metrics.accuracy_score(t_test[u_test==user], t_pred_u))'''

'''clf = svm.SVC(kernel='precomputed')
#linear kernel computation
gram_train = np.dot(np.array(X_train).transpose(), t_train)
clf.fit(gram_train, t_train)

gram_test = np.dot(np.array(X_test).transpose(), t_test)

t_pred_linear = clf.predict(gram_test)
print("Accuracy with linear kernel function:",metrics.accuracy_score(u_test, t_pred_linear))'''
