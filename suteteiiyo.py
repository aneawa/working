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

satellite = []
arrhythmia = []
ionosphere = []
annthyroid = []
pima = []
http=[]
cover=[]


ionosphere_auc = 0.845118632479
satellite_auc = 0.695031396538
arrhythmia_auc = 0.812499167499
annthyroid_auc = 0.797643479192
pima_auc = 0.676132354997
http_auc = 0.99988524389
cover_auc  = 0.840470272243

ionosphere_fnr = 0.11108249497
arrhythmia_fnr = 0.0739806763285
satellite_fnr = 0.122858629057
annthyroid_fnr = 0.0511107833061
pima_fnr = 0.158358373652
# http_fnr = 5.60354394649e-05
http_fnr = 5.60354394649*pow(10, -5)
cover_fnr = 0.0087520252075

ionosphere_fpr = 0.176804828974
arrhythmia_fpr = 0.0785797101449
satellite_fpr = 0.167248992902
annthyroid_fpr = 0.0505682680612
pima_fpr = 0.187412783295
http_fpr = 0.00021621134353
cover_fpr = 0.0117011497879

list_auc=[http_auc, cover_auc, satellite_auc, arrhythmia_auc, ionosphere_auc, annthyroid_auc, pima_auc]
list_fnr = [http_fnr, cover_fnr, satellite_fnr, arrhythmia_fnr, ionosphere_fnr, annthyroid_fnr, pima_fnr]
list_fpr = [http_fpr, cover_fpr, satellite_fpr, arrhythmia_fpr, ionosphere_fpr, annthyroid_fpr, pima_fpr]
list = [http, cover, satellite, arrhythmia, ionosphere, annthyroid, pima]
list2 = ["http", "forestcover", "satellite", "arrhythmia", "ionosphere", "annthyroid", "pima"]

# list_color = ["r", "b", "g", "c", "m"]
# list = [satellite_pr, arrhythmia_pr, ionosphere_pr, annthyroid_pr, pima_pr, http, cover]
# list2 = ["satellite", "arrhythmia", "ionosphere", "annthyroid", "pima", "http", "cover"]
list_color = ["r", "b", "g", "c", "m", "k", "y"]


f = open('/home/anegawa/デスクトップ/sotuken/results/subsamp/arrhy_size_fpr.txt', 'r')
for row in f:
    arrhythmia.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/subsamp/http_size_fpr.txt', 'r')
for row in f:
    http.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/subsamp/cover_size_fpr.txt', 'r')
for row in f:
    cover.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/subsamp/sate_size_fpr.txt', 'r')
for row in f:
    satellite.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/subsamp/iono_size_fpr.txt', 'r')
for row in f:
    ionosphere.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/subsamp/annth_size_fpr.txt', 'r')
for row in f:
    annthyroid.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/subsamp/pima_size_fpr.txt', 'r')
for row in f:
    pima.append(row[:-1])
f.close()

# hoge3 = []
# hoge4 = []
# for i in range(len(list)):
#     hoge3.append(np.argmax(list[i]))
#     hoge4.append(max(list[i]))
# print("index : " + str(hoge3))
# print("value : " + str(hoge4))
# print(list2)



plt.title("FPR by subsampling size")
xsize = 100
for i in range(len(list)):
    hoge = []
    hoge2 = []
    for j in range(len(list[i])):
        if j >= xsize:
            print("i : " +str(i))
            print('xsizeを超えましたよ〜')
            break
        if list[i][j] != 'nan':
            hoge.append(j+1)
            hoge2.append(list[i][j])
    print(len(hoge))
    print(len(hoge2))
            # plt.plot(range(1,101), list[i], label=list2[i], color=list_color[i])
            # plt.plot(list[i][j], label=list2[i], color=list_color[i])
            # plt.plot(j,list[i][j],'.' ,color=list_color[i])
    plt.plot(hoge, hoge2, label=list2[i], color=list_color[i])
    plt.hlines(list_fpr[i], 0, xsize, list_color[i], linestyles='dashed')  # hlines
plt.xlabel('subsampling size')
plt.ylabel('FPR score')
plt.ylim(0, 0.225)
# plt.xlim(0, 37)
plt.legend(loc = "center right")
# plt.legend()
plt.grid(True)
plt.show()

