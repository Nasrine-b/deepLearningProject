import h5py
import numpy as np
import random
import csv
from sklearn.model_selection import train_test_split

#%% DATA LOADING

dir_dataset_sg = 'SG24_dataset.h5'
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
    
ind_all = np.arange(X.shape[0])

ind_train, ind_test = train_test_split(ind_all,
                                       shuffle=True,
                                       stratify=T[ind_all],
                                       test_size=0.3,
                                       random_state=42)
ind_val, ind_test = train_test_split(ind_test,
                                     shuffle=True,
                                     stratify=T[ind_test],
                                     test_size=0.5,
                                       random_state=41)
## Set user 8 aside
user = 8
ind_train_u = ind_train[U[ind_train]==user] # indexes of samples of U8
ind_train = ind_train[U[ind_train]!=user] # remove U8 samples
# number of samples to replace in each set
n_train = ind_train_u.shape[0]
n_val   = n_train // 2
n_test  = n_train - n_val

ind_val = np.concatenate((ind_val, ind_train_u[:n_val])) # append u8 samples
ind_test = np.concatenate((ind_test, ind_train_u[-n_test:]))
ind_train = np.concatenate((ind_train, ind_val[:n_val], ind_test[:n_test]))
ind_val = ind_val[n_val:] # remove first n_val samples
ind_test = ind_test[n_test:] # remove first n_test samples

X_train = X[ind_train,:]
X_val   = X[ind_val,:]
X_test  = X[ind_test,:]
t_train = T[ind_train]
t_val   = T[ind_val]
t_test  = T[ind_test]
u_train = U[ind_train]
u_val   = U[ind_val]
u_test  = U[ind_test]

print(u_train==user)