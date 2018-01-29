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
list_auc=[http_auc, cover_auc, satellite_auc, arrhythmia_auc, ionosphere_auc, annthyroid_auc, pima_auc]
list = [http, cover, satellite, arrhythmia, ionosphere, annthyroid, pima]
list2 = ["http", "forestcover", "satellite", "arrhythmia", "ionosphere", "annthyroid", "pima"]

# list_color = ["r", "b", "g", "c", "m"]
# list = [satellite_pr, arrhythmia_pr, ionosphere_pr, annthyroid_pr, pima_pr, http, cover]
# list2 = ["satellite", "arrhythmia", "ionosphere", "annthyroid", "pima", "http", "cover"]
list_color = ["r", "b", "g", "c", "m", "k", "y"]


f = open('/home/anegawa/デスクトップ/sotuken/results/contamination/arrhy1.txt', 'r')
for row in f:
    arrhythmia.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/contamination/http1.txt', 'r')
for row in f:
    http.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/contamination/cover1.txt', 'r')
for row in f:
    cover.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/contamination/sate1.txt', 'r')
for row in f:
    satellite.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/contamination/iono1.txt', 'r')
for row in f:
    ionosphere.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/contamination/annth1.txt', 'r')
for row in f:
    annthyroid.append(row[:-1])
f.close()
f = open('/home/anegawa/デスクトップ/sotuken/results/contamination/pima1.txt', 'r')
for row in f:
    pima.append(row[:-1])
f.close()


plt.title("AUC by contamination")
xsize = 36
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
    plt.hlines(list_auc[i], 0, xsize, list_color[i], linestyles='dashed')  # hlines
plt.xlabel('contamination')
plt.ylabel('AUC score')
# plt.ylim(0, 0.225)
# plt.xlim(0, 37)
plt.legend()
plt.grid(True)
plt.show()

