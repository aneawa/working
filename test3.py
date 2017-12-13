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
arrhythmia_pr = []
ionosphere_pr = []
annthyroid_pr = []
pima_pr = []

list = [satellite_pr, arrhythmia_pr, ionosphere_pr, annthyroid_pr, pima_pr]
list2 = ["satellite", "arrhythmia", "ionosphere", "annthyroid", "pima"]
list_color = ["r", "b", "g", "c", "m"]


f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/arrhy_subsamp_fpr_100.txt', 'r')
for row in f:
    arrhythmia_pr.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/iono_subsamp_fpr_100.txt', 'r')
for row in f:
    ionosphere_pr.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/sate_subsamp_fpr_100.txt', 'r')
for row in f:
    satellite_pr.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/annth_subsamp_fpr_100.txt', 'r')
for row in f:
    annthyroid_pr.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/pima_subsamp_fpr_100.txt', 'r')
for row in f:
    pima_pr.append(row[:-1])
f.close()



ionosphere_fnr = 0.11108249497
arrhythmia_fnr = 0.0739806763285
satellite_fnr = 0.122858629057
annthyroid_fnr = 0.0511107833061
pima_fnr = 0.158358373652

ionosphere_fpr = 0.176804828974
arrhythmia_fpr = 0.0785797101449
satellite_fpr = 0.167248992902
annthyroid_fpr = 0.0505682680612
pima_fpr = 0.187412783295

ionosphere_auc = 0.845118632479
satellite_auc = 0.695031396538
arrhythmia_auc = 0.812499167499
annthyroid_auc = 0.797643479192
pima_auc = 0.676132354997

xx = []
for i in range(1,16):
    xx.append(2**i)
fnr = [satellite_fnr, arrhythmia_fnr, ionosphere_fnr, annthyroid_fnr, pima_fnr]
fpr = [satellite_fpr, arrhythmia_fpr, ionosphere_fpr, annthyroid_fpr, pima_fpr]
auc = [satellite_auc, arrhythmia_auc, ionosphere_auc, annthyroid_auc, pima_auc]
plt.figure(figsize=(10, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.2)
ax = plt.gca()
# ax.set_xscale('log')

# plt.suptitle("competition between pr-auc and roc-auc")

plt.hlines(y=fpr, xmin=0, xmax=100, colors=['r','b','g','c','m'], linestyles='dashed', linewidths=1)
# plt.hlines(y=auc, xmin=0, xmax=100, colors=['r','b','g','c','m'], linewidths=1)

plt.title("FPR by subsampling size")
for i in range(len(list)):
    hoge = []
    hoge2 = []
    for j in range(len(list[i])):
        if list[i][j] != 'nan':
            hoge.append(j)
            hoge2.append(list[i][j])

            # plt.plot(range(1,101), list[i], label=list2[i], color=list_color[i])
            # plt.plot(list[i][j], label=list2[i], color=list_color[i])
            # plt.plot(j,list[i][j],'.' ,color=list_color[i])
    plt.plot(hoge,hoge2, label=list2[i], color=list_color[i])
plt.xlabel('subsampling size')
plt.ylabel('FPR score')
plt.ylim(0, 0.225)
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
