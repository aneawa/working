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


satellite_pr = []
satellite_roc = []
arrhythmia_pr = []
arrhythmia_roc = []
ionosphere_pr = []
ionosphere_roc = []


list = [satellite_pr, satellite_roc, arrhythmia_pr, arrhythmia_roc, ionosphere_pr, ionosphere_roc]
list2 = ["satellite_pr", "satellite_roc", "arrhythmia_pr", "arrhythmia_roc", "ionosphere_pr", "ionosphere_roc"]
list_color = ["r", "r", "b", "b", "g", "g"]

f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/arrhy_pr_tree.txt', 'r')
for row in f:
    arrhythmia_pr.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/arrhy_roc_tree.txt', 'r')
for row in f:
    arrhythmia_roc.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/sate_pr_tree.txt', 'r')
for row in f:
    satellite_pr.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/sate_roc_tree.txt', 'r')
for row in f:
    satellite_roc.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/iono_pr_tree.txt', 'r')
for row in f:
    ionosphere_pr.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/newest/base_data/iono_roc_tree.txt', 'r')
for row in f:
    ionosphere_roc.append(row[:-1])
f.close()


plt.figure(figsize=(10, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.2)
plt.suptitle("competition between pr-auc and roc-auc")



plt.subplot(1, 2, 1)
plt.plot(arrhythmia_pr, label="arrhythmia", color="r")
plt.plot(ionosphere_pr, label="ionosphere", color="b")
plt.plot(satellite_pr, label="satellite", color="g")
plt.xlabel('number of trees')
plt.ylabel('pr-auc score')
plt.ylim(0.6, 1)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(arrhythmia_roc, label="arrhythmia", color="r")
plt.plot(ionosphere_roc, label="ionosphere", color="b")
plt.plot(satellite_roc, label="satellite", color="g")
plt.xlabel('number of trees')
plt.ylabel('roc-auc score')
plt.ylim(0.6, 1)
plt.legend()
# plt.tight_layout()

plt.show()

# for i in range(len(list)):
#     if i%2==0:
#         plt.plot(list[i], label=list2[i], color=list_color[i])
#     else:
#         plt.plot(list[i], label=list2[i], color=list_color[i], linestyle="dashed")
# plt.xlabel('number of trees')
# plt.ylabel('auc score')
# plt.legend()
# plt.show()
