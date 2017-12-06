import sys
import math
import numpy as np
import random
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


satellite_pr = []
satellite_roc = []
arrhythmia_pr = []
arrhythmia_roc = []
ionosphere_pr = []
ionosphere_roc = []


# list = [satellite_pr, satellite_roc, arrhythmia_pr, arrhythmia_roc, ionosphere_pr, ionosphere_roc]
# list2 = ["satellite_pr", "satellite_roc", "arrhythmia_pr", "arrhythmia_roc", "ionosphere_pr", "ionosphere_roc"]
# list_color = ["r", "r", "b", "b", "g", "g"]
list = [satellite_pr, arrhythmia_pr, ionosphere_pr]
list2 = ["satellite", "arrhythmia", "ionosphere"]
list_color = ["r", "b", "g"]


f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/arrhy_subsamp_log.txt', 'r')
for row in f:
    arrhythmia_pr.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/iono_subsamp_log.txt', 'r')
for row in f:
    ionosphere_pr.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/sate_subsamp_log.txt', 'r')
for row in f:
    satellite_pr.append(row[:-1])
f.close()


ionosphere_plain = 0.175947686117
arrhythmia_plain = 0.0767826086957
satellite_plain = 0.166598293968

ionosphere_cri = 0.84241982906
satellite_cri = 0.682251407222
arrhythmia_cri = 0.812402597403

xx = []
for i in range(1,16):
    xx.append(2**i)
cri = [satellite_cri, arrhythmia_cri, ionosphere_cri]
plains = [satellite_plain, arrhythmia_plain, ionosphere_plain]
plt.figure(figsize=(10, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.2)
ax = plt.gca()
ax.set_xscale('log')

# plt.suptitle("competition between pr-auc and roc-auc")

# plt.hlines(y=plains, xmin=0, xmax=40000, colors=['r','b','g'], linestyles='dashed', linewidths=1)
plt.hlines(y=cri, xmin=0, xmax=40000, colors=['r','b','g'], linestyles='dashed', linewidths=1)

plt.title("AUC by subsampling size")
for i in range(len(list)):
    plt.plot(xx, list[i], label=list2[i], color=list_color[i])
plt.xlabel('subsumpling size')
plt.ylabel('AUC score')
# plt.ylim(0, 0.2)
plt.legend()
plt.grid(True)
plt.show()


# plt.subplot(1, 2, 1)
# plt.plot(arrhythmia_pr, label="arrhythmia", color="r")
# plt.plot(ionosphere_pr, label="ionosphere", color="b")
# plt.plot(satellite_pr, label="satellite", color="g")
# plt.xlabel('number of trees')
# plt.ylabel('pr-auc score')
# plt.ylim(0.6, 1)
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(arrhythmia_roc, label="arrhythmia", color="r")
# plt.plot(ionosphere_roc, label="ionosphere", color="b")
# plt.plot(satellite_roc, label="satellite", color="g")
# plt.xlabel('number of trees')
# plt.ylabel('roc-auc score')
# plt.ylim(0.6, 1)
# plt.legend()
#
# plt.show()


# nn_anomaly = 126
# nn_normal = 225
# cross_count = 5
# contamination = 0.1
# for a in range(10):
#     print("-----------------------------")
#     print(a)
#     anomaly_rate = a/100
#     for count in range(cross_count):
#         for i in range(cross_count):
#             print(count, i)
#             if i != count:
#                 train_flag = True
#                 if anomaly_rate is not None:
#                     if contamination != anomaly_rate:
#                         train_flag = False
#                         if contamination > anomaly_rate:  # 異常系を減らす
#                             k = nn_anomaly * (anomaly_rate / (1 - anomaly_rate))
#                             # k = int(nn_anomaly * (anomaly_rate / (1 - anomaly_rate)))
#                             # k = int(np.ceil(nn_anomaly * (anomaly_rate / (1 - anomaly_rate))))
#                         else:  # 正常系を減らす
#                             n_normal = np.ceil(nn_anomaly / anomaly_rate) - nn_anomaly
#                             # n_normal = int(nn_anomaly / anomaly_rate) - nn_anomaly
#                             normal_rate = n_normal / nn_normal
#                             # k = int(nn_normal * normal_rate)
#                             k = int(np.ceil(nn_normal * normal_rate))
#                         print("train_anomaly_rate")
#                         print("k : " + str(k))
#
#                 if train_flag:
#                     print("train")
#
#             else:
#                 print("test")
