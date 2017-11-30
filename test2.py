import sys
import math
import numpy as np
import pandas as pd
import datetime
import scipy.io
import h5py
import glob
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import time

rate = 0.25
cutter = 6435 * rate
cutter1 = 2036 * rate
cutter2 = 4399 * rate
count = int(1/rate)
print(count)

for i in range(count):
    head = int(cutter1 * i)
    tail = int(cutter1 * (i + 1))-1
    print(head, tail, tail - head + 1, 11111111)

    head2 = int(cutter2 * i)
    tail2 = int(cutter2 * (i + 1)) - 1
    print(head2, tail2, tail2 - head2 + 1, 22222222)
    # X_sepa.append(X[head:tail])

# for count in range(5):
#     part = int(np.ceil(cutter * count))
#     for i,k in zip(range(19998), range(part, part+19998)):
#         while k >= 19998:
#             k = k - 19998
#         if i == int(cutter):
#             print("train end : " + str(i) + ", " +  str(k))
#         if i == 0:
#             print(i, k, 10000)
#         if k == 19997:
#             print(i, k, 20000)
#         if k == 0:
#             print(i, k, 30000)
#         if i == 19997:
#             print(i, k, 40000)
#     print("")


# def kurtosis_skewness(data):
#     data = np.array(data)
#     ave = np.average(data)
#     std = np.std(data)
#     kurtosis = np.average((data - ave)**3) / (std**3)
#     skewness = (np.average((data - ave)**4) / (std**4)) - 3
#     return kurtosis, skewness
#
# data = [1,2,3,4,5]
# [kur, ske] = kurtosis_skewness(data)
# print(kur, ske)

# filename = '/home/anegawa/Dropbox/arrhythmia.mat'
# sepaLabel = True
#
# mat = scipy.io.loadmat(filename)
# X = mat['X']
# y = mat['y']
#
# pca2 = PCA(copy=True, iterated_power='auto', random_state=None,
#            svd_solver='auto', tol=0.0, whiten=False)
# pca2.fit(X)
# X_after = pca2.transform(X)
# # print(X_after.shape)
# # X = X_after
#
# bunsan = []
# for i in range(X.shape[1]):
#     a = []
#     for j in range(len(X)):
#         # print(i, j)
#         a.append(X[j][i])
#         # print(len(a))
#     b = np.var(a)
#     bunsan.append(b)
#     # print(b)
#
# bunsan3 = []
# for i in range(X_after.shape[1]):
#     a = []
#     for j in range(len(X_after)):
#         # print(i, j)
#         a.append(X_after[j][i])
#         # print(len(a))
#     b = np.var(a)
#     bunsan3.append(b)
# #次元ごとの分散を計算しましょうね〜〜
#
#
# # print(bunsan)
#
# min = 0
# max = 6061.45844705
# # for i in range(len(bunsan)):
# #     a = []
# #     if min > bunsan[i]:
# #         min = bunsan[i]
# #     if max < bunsan[i]:
# #         max = bunsan[i]
# #ここまででminとmaxの計算おわおわり
#
# a = []
# aa = []
# for i in range(len(bunsan)):
#     if (max - min) != 0:
#         z = (bunsan[i] - min)/(max - min)
#         w = (bunsan3[i] - min)/(max - min)
#     else :
#         z = 0
#         w = 0
#     # print(z)
#     a.append(z)
#     aa.append(w)
# bunsan2 = np.array(a)
# bunsan4 = np.array(aa)
# #正規化おわおわり
#
# a = False
# if a:
#     buf = 0
#     sorted = []
#     for i in range(len(bunsan2)):
#         buf += bunsan2[i]
#         sorted.append(buf)
# else:
#     sorted = np.sort(bunsan2)[::-1]
#     sorted2 = np.sort(bunsan4)[::-1]
#     for i in range(len(sorted)):
#         print(sorted[i])
#         print(sorted2[i])
#
#
#
# #ついでにプロットしましょうか
# xlim_limit = 33
#
# plt.figure(figsize=(8, 10))
# plt.suptitle("arrhythmia")
# plt.subplots_adjust(wspace=0.5, hspace=0.4)
# plt.subplot(3, 1, 1)
# plt.bar(range(len(sorted)), sorted, align = 'center', color = "r")
# # plt.plot(moto,yy, ".")
# plt.title('var of original dim')
# plt.ylim(0, 1)
# plt.xlim(0, xlim_limit)
# plt.grid(True)
# plt.ylabel('var')
# plt.xlabel('dim')
#
# plt.subplot(3, 1, 2)
# plt.bar(range(len(sorted2)), sorted2, align = 'center', color = "g")
# # plt.plot(moto,yy, ".")
# plt.title('var of pca dim')
# plt.ylim(0, 1)
# plt.xlim(0, xlim_limit)
# plt.grid(True)
# plt.ylabel('var')
# plt.xlabel('dim')
#
#
# plt.subplot(3, 1, 3)
# sample = []
# j = 0
# for file in glob.glob('/home/anegawa/デスクトップ/sotuken/results/newer/pca/arrhy_tree2_sfl.txt'):
#     # print(j)
#     print(file)
#     for line in open(file, 'r'):
#         sample.append(float(line[:-1]))
#         # print(line)
#     j = j + 1
#
# ionosphere3 = 0.869665551839
# satellite3 = 0.67975921935
# arrhythmia3 = 0.814845154845
# plt.plot(sample, label='arrhythmia_pca', color = "r")
# plt.hlines(y=arrhythmia3, xmin=0, xmax=len(sample), colors='b', linestyles='dashed', linewidths=1)
# plt.title("AUC of numbers of trees")
# plt.xlabel('number of trees')
# plt.ylabel('AUC')
# # plt.xlim(0, 300)
#
# plt.legend()
# plt.grid(True)
# # plt.tight_layout()
#
# plt.show()