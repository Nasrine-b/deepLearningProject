import h5py
import numpy as np
import random
import csv
from sklearn.model_selection import train_test_split
from collections import Counter
#%% DATA LOADING

dir_dataset_sg = 'data/SG24_dataset.h5'
#dir_dataset_sg = './dataset/SG24_dataset_euler.h5'

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


#%% SET SPLITTING (TRAIN + VALIDATION + TEST)
## Data splitting 1 : all -> train and rest
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=123)
#
#for train_index, test_index in sss.split(X,T,groups=U):
#    X_train, X_test = X[train_index], X[test_index]
#    t_train, t_test = T[train_index], T[test_index]
#    u_train, u_test = U[train_index], U[test_index]
#    ind_train, ind_test = train_index, test_index
#
## Data splitting 2 : test -> validation and test
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=321)
#
#for val_index, test_index in sss.split(X_test, t_test):
#    X_val, X_test = X_test[val_index], X_test[test_index]
#    t_val, t_test = t_test[val_index], t_test[test_index]
#    u_val, u_test = u_test[val_index], u_test[test_index]
#    ind_val, ind_test = ind_test[val_index], ind_test[val_index]
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
# number of samples to replace in each set



#ind_val = np.concatenate((ind_val, ind_train_u[:n_val])) # append u8 samples
"""
tab_val = ind_val[U[ind_val]!=user]
tab_test = ind_test[U[ind_test]!=user]
ind_train = np.concatenate((ind_train, tab_val[:n_val], tab_test[:n_test]))


#ind_val = ind_val[ind_val!=tab_val[:n_val]]
c_val = Counter(ind_val)
c_tab_val = Counter(tab_val[n_val:])
ind_val = sorted((c_val - c_tab_val).elements())
ind_val=np.array(ind_val)

c_test = Counter(ind_test)
c_tab_test = Counter(tab_test[n_test:])
ind_test = sorted((c_val - c_tab_test).elements())
ind_test=np.array(ind_test)
"""
print(ind_train.shape[0])
print(ind_test.shape[0])
print(ind_val.shape[0])
X_train = X[ind_train,:]
X_val   = X[ind_val,:]
X_test  = X[ind_test,:]
t_train = T[ind_train]
t_val   = T[ind_val]
t_test  = T[ind_test]
u_train = U[ind_train]
u_val   = U[ind_val]
u_test  = U[ind_test]

for u in np.unique(u_val):
    print('User %i: %i samples out of total %i (%.1f%%)' % (u, sum(u_val==u), len(u_val), sum(u_val==u)/len(u_val)*100))
print(X_test[u_test==8])
np.set_printoptions(threshold=np.inf)
