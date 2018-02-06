import sys
import math
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn import metrics

import scipy.io
import h5py

name = "ionosphere"
tit = "variance of " + name
filename = '/home/anegawa/Dropbox/' + name + '.mat'

if filename == '/home/anegawa/Dropbox/http.mat' or filename == '/home/anegawa/Dropbox/smtp.mat':
    mat = {}
    f = h5py.File(filename)
    for k, v in f.items():
        mat[k] = np.array(v)
    X = mat['X'].T
    y2 = mat['y'][0]
    y3 = []
    for i in range(len(y2)):
        y3.append(int(y2[i]))
    y = np.reshape(y3, [len(y3), 1])
else:
    mat = scipy.io.loadmat(filename)
    X = mat['X']
    y = mat['y']

X_anomaly = []
X_normal = []
for i in range(len(X)):
    if y[i] == 1:
        X_anomaly.append(X[i])
    else:
        X_normal.append(X[i].tolist())

var = []
for i in range(len(X[0])):
    data = np.array(X[:,i])
    var.append(np.var(data))

var_normal = []
for i in range(len(X_normal[0])):
    hoge = []
    for j in range(len(X_normal)):
        hoge.append(X_normal[j][i])
    data = np.array(hoge)
    # data = X_normal[:,i]
    var_normal.append(np.var(data))
    # var_normal.append(np.var(X_normal[:,i]))

print(var)
print(var_normal)


plt.figure(figsize=(8, 8))
plt.subplots_adjust(wspace=0.5, hspace=0.6)
plt.subplot(3, 1, 1)
plt.suptitle(tit)
# plt.bar(range(len(var)), np.sort(var)[::-1], align = 'center', color = "r")
plt.bar(range(len(var)), var, align = 'center', color = "r")

# plt.bar(range(len(var_pca_rank)), np.sort(var_pca_rank)[::-1], align = 'center', color = "r")

# for i in range(len(var_pca_rank)):
#     plt.bar(i, var_pca[var_pca_rank[i]], align = 'center', color = "r")
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
# plt.plot(moto,yy, ".")
plt.grid(True)
plt.ylabel('var')
plt.xlabel('dim')
plt.title("original")


plt.subplot(3, 1, 2)
# plt.bar(range(len(var_pca_rank)), np.sort(var_pca)[::-1], align = 'center', color = "g")
plt.bar(range(len(var_normal)), var_normal, align = 'center', color = "g")

# for i in range(len(var_rank)):
#     plt.bar(i, var[var_rank[i]], align = 'center', color = "g")
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
# plt.plot(moto,yy, ".")
plt.grid(True)
plt.ylabel('var')
plt.xlabel('dim')
plt.title("normal")


plt.subplot(3, 1, 3)
hoge2 = []
for i in range(len(var)):
    hoge2.append(var[i] - var_normal[i])
# hoge2[4] = -5
print(hoge2)
# plt.bar(range(len(var_pca_rank)), np.sort(var_pca)[::-1], align = 'center', color = "g")
plt.bar(range(len(hoge2)), hoge2, align = 'center', color = "b")

# for i in range(len(var_rank)):
#     plt.bar(i, var[var_rank[i]], align = 'center', color = "g")
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
# plt.plot(moto,yy, ".")
plt.grid(True)
# plt.ylim(-0.0005,0.0005)
plt.ylabel('var')
plt.xlabel('dim')
plt.title("original - normal")

# plt.tight_layout()
# plt.grid(True)
plt.legend()
plt.show()


