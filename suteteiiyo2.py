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

name = "http"
tit = ""


# filename = '/home/anegawa/デスクトップ/sotuken/results/var/' + name + '_pca.txt'
# filename2 = '/home/anegawa/デスクトップ/sotuken/results/var/' + name + '_ori.txt'
filename3 = '/home/anegawa/デスクトップ/sotuken/results/other/sate_sizechange_time.txt'
filename4 = '/home/anegawa/デスクトップ/sotuken/results/other/annth_sizechange_time.txt'

# var = []
# var_pca = []
# f = open(filename2, 'r')
# for row in f:
#     var.append(float(row[:-1]))
# f.close()
# f = open(filename, 'r')
# for row in f:
#     var_pca.append(float(row[:-1]))
# f.close()

sate = []
f = open(filename3, 'r')
for row in f:
    sate.append(float(row[:-1]))
f.close()
annth = []
f = open(filename4, 'r')
for row in f:
    annth.append(float(row[:-1]))
f.close()


# x = []
# for i in range(len(var)):
#     x.append(i)

## var_rank = np.argsort(var)[::-1]
## var_pca_rank = np.argsort(var_pca)[::-1]
# var_rank = np.sort(var)[::-1]
# var_pca_rank = np.sort(var_pca)[::-1]

# hoge = []
# hoge2 = 0
# for i in range(len(var_rank)):
#     hoge2 += var_rank[i]
#     hoge.append(hoge2)
# var_rank = hoge


# hoge = []
# hoge2 = 0
# for i in range(len(var_pca_rank)):
#     hoge2 += var_pca_rank[i]
#     hoge.append(hoge2)
# var_pca_rank = hoge



# print(var_rank)
# print(var_pca_rank)
## print(var_pca[var_pca_rank[0]])

plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.subplot(2, 1, 1)
# plt.suptitle(tit)
## plt.bar(range(len(var)), np.sort(var)[::-1], align = 'center', color = "r")
# plt.bar(range(len(var)), var_rank, align = 'center', color = "r")
hoge = []
for i in range(0, len(sate)):
    hoge.append(i*100)
plt.plot(hoge, sate,label="satellite")
hoge = []
for i in range(0, len(annth)):
    hoge.append(i*100)
plt.plot(hoge, annth,label="annthyroid")

## plt.bar(range(len(var_pca_rank)), np.sort(var_pca_rank)[::-1], align = 'center', color = "r")

# for i in range(len(var_pca_rank)):
#     plt.bar(i, var_pca[var_pca_rank[i]], align = 'center', color = "r")
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
# plt.plot(moto,yy, ".")
plt.grid(True)
plt.legend()
plt.ylabel('time(s)')
plt.xlabel('data size')
plt.title("nonsubsampling time")


# plt.subplot(2, 1, 2)




# # plt.bar(range(len(var_pca_rank)), np.sort(var_pca)[::-1], align = 'center', color = "g")
# plt.bar(range(len(var_pca_rank)), var_pca_rank, align = 'center', color = "g")
#
# # for i in range(len(var_rank)):
# #     plt.bar(i, var[var_rank[i]], align = 'center', color = "g")
# plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
# # plt.plot(moto,yy, ".")
# plt.grid(True)
# plt.ylabel('time(s)')
# plt.xlabel('data size')
# plt.title("Annthyroid")

# plt.tight_layout()
# plt.grid(True)
plt.legend()
plt.show()