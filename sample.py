import sys
import math
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn import metrics

import scipy.io
import h5py

# data cut

len_anomaly = 2036
len_normal = 4399
rate = 0.2
cross_count = 5
contamination = 0.36
anomaly_rate = 0.66


cutter_anomaly = len_anomaly * rate
cutter_normal = len_normal * rate
X_sepa_ano = []
X_sepa_nor = []
for i in range(cross_count):
    head2 = int(cutter_normal * i)
    tail2 = int(cutter_normal * (i + 1)) - 1
    # X_sepa_nor.append(X_normal[head2:tail2 + 1])
    print("head2 : " + str(head2))
    print("tail2 : " + str(tail2))
    print(tail2 - head2 + 1)
    X_sepa_nor.append(tail2 - head2 + 1)

    head = int(cutter_anomaly * i)
    tail = int(cutter_anomaly * (i + 1)) - 1
    # X_sepa_ano.append(X_anomaly[head:tail + 1])
    print("head : " + str(head))
    print("tail : " + str(tail))
    print(tail - head + 1)
    X_sepa_ano.append(tail - head + 1)

    print("----------------------")

for count in range(cross_count):
    if True:
        X_train = []
        X_train_correct = []
        X_test = []
        X_test_correct = []

        # 学習データの異常系含有率を変更
        for i in range(cross_count):
            if i != count:
                print("トレーニング")
                print("i : " + str(i))
                print("count : " + str(count))
                train_flag = True
                if anomaly_rate is not None:
                    if contamination != anomaly_rate:
                        train_flag = False
                        if contamination > anomaly_rate:  # 異常系を減らす
                            k = int(np.ceil(X_sepa_nor[i] * (anomaly_rate / (1 - anomaly_rate))))
                            print("異常系を以下の量に減らします")
                            print("k : " + str(k))

                            # anomaly_hoge = random.sample(X_sepa_ano[i], k)  # ランダムに抽出
                            # normal_hoge = X_sepa_nor[i]

                        else:  # 正常系を減らす
                            n_normal = int(X_sepa_ano[i] / anomaly_rate) - X_sepa_ano[i]
                            normal_rate = n_normal / X_sepa_nor[i]
                            k = int(np.ceil(X_sepa_nor[i] * normal_rate))
                            print("正常系を以下の量に減らします")
                            print("k : " + str(k))

                            # normal_hoge = random.sample(X_sepa_nor[i], k)  # ランダムに抽出
                            # anomaly_hoge = X_sepa_ano[i]
                        print("anomaly_rateを変えてモデリング")
                        # X_train.extend(anomaly_hoge)
                        # for j in range(len(anomaly_hoge)):
                        #     X_train_correct.append(-1)
                        #
                        # X_train.extend(normal_hoge)
                        # for j in range(len(normal_hoge)):
                        #     X_train_correct.append(1)

                if train_flag:
                    # X_train.extend(X_sepa_ano[i])
                    # for j in range(len(X_sepa_ano[i])):
                    #     X_train_correct.append(-1)
                    #
                    # X_train.extend(X_sepa_nor[i])
                    # for j in range(len(X_sepa_nor[i])):
                    #     X_train_correct.append(1)
                    print("普通にモデリング")
            else:
                print("テスト")
                print("i : " + str(i))
                print("count : " + str(count))
            print("--------------------")