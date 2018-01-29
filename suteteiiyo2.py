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

tit = "variance of ForestCover"
filename = '/home/anegawa/デスクトップ/sotuken/results/var/cover_pca.txt'
filename2 = '/home/anegawa/デスクトップ/sotuken/results/var/cover_ori.txt'

var = []
var_pca = []
f = open(filename2, 'r')
for row in f:
    var.append(row[:-1])
f.close()
f = open(filename, 'r')
for row in f:
    var_pca.append(row[:-1])
f.close()

x = []
for i in range(len(var)):
    x.append(i)

# var_rank = np.argsort(var)[::-1]
# var_pca_rank = np.argsort(var_pca)[::-1]
var_rank = np.sort(var)[::-1]
var_pca_rank = np.sort(var_pca)[::-1]

print(var_rank)
print(var_pca_rank)
# print(var_pca[var_pca_rank[0]])

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.subplot(2, 1, 1)
plt.suptitle(tit)
# plt.bar(range(len(var)), np.sort(var)[::-1], align = 'center', color = "r")
plt.bar(range(len(var_pca_rank)), np.sort(var_pca_rank)[::-1], align = 'center', color = "r")
# for i in range(len(var_pca_rank)):
#     plt.bar(i, var_pca[var_pca_rank[i]], align = 'center', color = "r")
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
# plt.plot(moto,yy, ".")
plt.grid(True)
plt.ylabel('var')
plt.xlabel('dim')
plt.title("pca")


plt.subplot(2, 1, 2)
plt.bar(range(len(var_rank)), var_rank, align = 'center', color = "g")
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
# plt.plot(moto,yy, ".")
plt.grid(True)
plt.ylabel('var')
plt.xlabel('dim')
plt.title("original")

# plt.tight_layout()
# plt.grid(True)
plt.legend()
plt.show()