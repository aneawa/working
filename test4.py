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

tit = "mammaography"
filename = '/home/anegawa/Dropbox/mammography.mat'

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

kurtosis_rank = np.argsort(kurtosis_set)
skewness_rank = np.argsort(skewness_set)
flag_a = True
flag_n = True

for i in range(len(X)):
    if y[i] == 1: #異常
        if flag_a:
            plt.plot(X[i][kurtosis_rank[0]], X[i][kurtosis_rank[1]], '.', color="r", label="anomaly")
            flag_a = False
        else:
            plt.plot(X[i][kurtosis_rank[0]], X[i][kurtosis_rank[1]], '.', color="r")
    else: #正常
        if flag_n:
            plt.plot(X[i][kurtosis_rank[0]], X[i][kurtosis_rank[1]], '.', color="b", label = "normal")
            flag_n = False
        else:
            plt.plot(X[i][kurtosis_rank[0]], X[i][kurtosis_rank[1]], '.', color="b")

# for i in range(len(X)):
#     plt.plot(X[i][kurtosis_rank[0]], X[i][kurtosis_rank[1]], '.', color = "r")
#     print(X[i][kurtosis_rank[0]], X[i][kurtosis_rank[1]])
plt.title(tit)
plt.grid(True)
plt.legend()
plt.show()