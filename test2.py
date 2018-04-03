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

satellite = []
annthyroid = []
arrhythmia = []
mammography = []
pima = []
ionosphere = []

list = [satellite, arrhythmia, mammography, pima, ionosphere]
list2 = ["satellite", "arrhythmia", "mammography", "pima", "ionosphere"]
f = open('C:\Users\Riku Anegawa\Desktop/デスクトップ/sotuken/results/newest/base_data/arrhy_cont.txt', 'r')
for row in f:
    arrhythmia.append(row[:-1])
f.close()
f = open('C:\Users\Riku Anegawa\Desktop/デスクトップ/sotuken/results/newest/base_data/pima_cont.txt', 'r')
for row in f:
    pima.append(row[:-1])
f.close()
f = open('C:\Users\Riku Anegawa\Desktop/デスクトップ/sotuken/results/newest/base_data/mammo_cont.txt', 'r')
for row in f:
    mammography.append(row[:-1])
f.close()
f = open('C:\Users\Riku Anegawa\Desktop/デスクトップ/sotuken/results/newest/base_data/sate_cont.txt', 'r')
for row in f:
    satellite.append(row[:-1])
f.close()
f = open('C:\Users\Riku Anegawa\Desktop/デスクトップ/sotuken/results/newest/base_data/iono_cont.txt', 'r')
for row in f:
    ionosphere.append(row[:-1])
f.close()

var=[]
kurto = []
list = [var, kurto]
list2 = ["variance", "kurtosis"]


# #ついでにプロットしましょうか
#
# plt.figure(figsize=(8, 10))
plt.title("auc score(change contamination)")
for i in range(len(list)):
    plt.plot(list[i], label=list2[i])
plt.xlabel('contamination')
plt.ylabel('auc')
plt.legend()
plt.grid(True)
plt.show()

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
# for file in glob.glob('C:\Users\Riku Anegawa\Desktop/デスクトップ/sotuken/results/newer/pca/arrhy_tree2_sfl.txt'):
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