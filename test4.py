import sys
import math
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn import metrics

import scipy.io
import h5py

tit = "kurtosis and variance of annthyroid"
filename = '/home/anegawa/Dropbox/annthyroid.mat'

mat = scipy.io.loadmat(filename)
X = mat['X']
y = mat['y']

kurtosis_set = []
skewness_set = []
var = []

for i in range(len(X[0])):
    data = []
    for j in range(len(X)):
        data.append(X[j][i])
    data = np.array(data)
    ave = np.average(data)
    std = np.std(data)
    kurtosis = np.average((data - ave)**3) / (std**3)
    skewness = np.average((data - ave)**4) / (std**4) - 3
    # if math.isnan(kurtosis):
    #     print("-----------------------")
    #     print("data : " + str(data))
    #     print("ave : " + str(ave))
    #     print("std : " + str(std))
    # print("kurtosis : " + str(kurtosis))
    # print("skewness : " + str(skewness))
    var.append(std)
    kurtosis_set.append(kurtosis)
    skewness_set.append(skewness)


var_rank = np.argsort(var)[::-1]
kurtosis_rank = np.argsort(kurtosis_set)[::-1]
skewness_rank = np.argsort(skewness_set)[::-1]


# flag_a = True
# flag_n = True
# for i in range(len(X)):
#     if y[i] == 1: #異常
#         if flag_a:
#             plt.plot(X[i][kurtosis_rank[0]], X[i][kurtosis_rank[1]], '.', color="r", label="anomaly")
#             flag_a = False
#         else:
#             plt.plot(X[i][kurtosis_rank[0]], X[i][kurtosis_rank[1]], '.', color="r")
#     else: #正常
#         if flag_n:
#             plt.plot(X[i][kurtosis_rank[0]], X[i][kurtosis_rank[1]], '.', color="b", label = "normal")
#             flag_n = False
#         else:
#             plt.plot(X[i][var_rank[0]], X[i][var_rank[1]], '.', color="b")




plt.subplots_adjust(wspace=0.5, hspace=0.4)
plt.subplot(2, 1, 1)
plt.suptitle(tit)
plt.bar(range(len(var)), np.sort(var)[::-1], align = 'center', color = "r")
# plt.plot(moto,yy, ".")
plt.grid(True)
plt.ylabel('variance')
plt.xlabel('dim')

plt.subplot(2, 1, 2)
for i in range(len(kurtosis_set)):
    plt.bar(i, kurtosis_set[i], align = 'center', color = "g")
# plt.plot(moto,yy, ".")
plt.grid(True)
plt.ylabel('kurtosis')
plt.xlabel('dim')

plt.grid(True)
plt.legend()
plt.show()