import sys
import numpy as np
import pandas as pd
import datetime
import scipy.io
import h5py
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import time
import re


filename = '/home/anegawa/Dropbox/ionosphere.mat'
sepaLabel = True

f = open('/home/anegawa/デスクトップ/sotuken/results/time/satellite_time2.txt', 'r')      #hoge
fp = open('/home/anegawa/デスクトップ/sotuken/results/time/satellite_time_pca2.txt', 'r') #hogg
diff = False
tit = "satellite"

j = 0
hoge = [] #ファイルの中身そのまま
hogg = []
for row, row2 in zip(f, fp):
    a = row[1:-2]
    b = row2[1:-2]
    hoge.append(a)
    hogg.append(b)
    j = j + 1
f.close()
fp.close()


hoge3 = [] #スペースで分割して配列化
hogg3 = []
for i in range(len(hoge)):
    hoge2 = re.split(" +", hoge[i])
    hogg2 = re.split(" +", hogg[i])

    hoge2 = np.array(hoge2)
    hogg2 = np.array(hogg2)

    j = 0
    end = len(hoge2)
    while j < end:
        # print("---------")
        # print(hoge2)
        # print(hogg2)
        # print(j, end, len(hoge2))

        flg = 0
        if j < len(hoge2):
            if hoge2[j] == '':
                hoge2 = np.delete(hoge2, j, 0)
                flg = 1
        if j < len(hogg2):
            if hogg2[j] == '':
                hogg2 = np.delete(hogg2, j, 0)
                flg = 1
        if flg == 1:
            A = max(len(hoge2), len(hogg2))
            end = A
        j += 1


    hogehoge = []
    hogghogg = []
    for j in range(len(hoge2)):
        hogehoge.append(float(hoge2[j]))
        hogghogg.append(float(hogg2[j]))
    hoge3.append(hogehoge)
    hogg3.append(hogghogg)


all = []
pca_train = []
fit = []
pca_test = []
test = []
train_sum = []
test_sum = []
list = [all,pca_train, fit, pca_test, test, train_sum, test_sum]
list_name = ["all", "pca_train", "fit", "pca_test", "test", "train_sum", "test_sum"]
all2 = []
pca_train2 = []
fit2 = []
pca_test2 = []
test2 = []
train_sum2 = []
test_sum2 = []
list2 = [all2, pca_train2, fit2, pca_test2, test2, train_sum2, test_sum2]
# list_name = ["all", "pca_train", "fit", "pca_test", "test", "train_sum", "test_sum"]


for i in range(len(hoge3[0])):
    for j in range(len(hoge3)):
        list[i].append(hoge3[j][i])
        list2[i].append(hogg3[j][i])

if diff:
    ane = np.array(list)
    ane2 = np.array(list2)
    list = ane2 - ane

x = np.array([10.,20.,30.,40.,50.,60.,70.,80.,90.,100.])
# all = np.array(all)
# for i in range(len(all)):
#     plt.plot(x[i],all[i])
#
c = ["r", "g", "b", "black", "y", "c", "m"]
y = [0.5, 0.4, 0.13, 0.035, 0.00163]
plt.figure(figsize = (100, 200))
plt.suptitle(tit)
plt.subplots_adjust(wspace=0.5, hspace=0.2)
m = 4
for i in range(7):
    # if i == 0:
    if True:
        # plt.subplot(2, 5, i*2+1)
        plt.subplot(2, m, i+1)
        plt.title(list_name[i])
        if diff:
           plt.plot(x, list[i], marker="D", color = c[i])
        else:
            plt.plot(x, list[i], label="original", marker="D", color = "r")
            plt.plot(x, list2[i], label="pca", marker="D", color = "b")
        plt.grid(True)
        plt.xlabel('training rate')
        plt.ylabel('times')
        plt.xlim(0, 110)
        # plt.ylim(0, y[i])
        plt.legend()

        # plt.subplot(2, m, i+1)
        # plt.plot(x, list2[i], label=list2_name[i], marker="D", color = c[i])
        # plt.grid(True)
        # plt.xlabel('training rate')
        # plt.ylabel('times')
        # plt.xlim(0, 110)
        # plt.ylim(0, y[i])
        # plt.legend()
    elif i == 1:
        unko = 0
    else:
        # plt.subplot(2, 5, i*2+1)
        plt.subplot(2, m, (i-1)+1)
        plt.title(list_name[i])
        plt.plot(x, list[i], label=list_name[i], marker="D", color = c[i-1])
        plt.grid(True)
        plt.xlabel('training rate')
        plt.ylabel('times')
        plt.xlim(0, 110)
        plt.legend()

        # plt.subplot(2, 5, i*2+2)

        plt.subplot(2, m, (i-1)+5)
        plt.plot(x, list2[i], label=list2_name[i], marker="D", color = c[i-1])
        plt.grid(True)
        plt.xlabel('training rate')
        plt.ylabel('times')
        plt.xlim(0, 110)
        plt.legend()



# plt.tight_layout()
plt.show()