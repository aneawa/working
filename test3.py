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


contamination = 0.32
all = 6435
len_anomaly = 2036
len_normal = 4399
for i in range(101):
    anomaly_rate = i / 100
    if contamination == anomaly_rate:
        print("i : " + str(i) + "\tc\t" + str(len_anomaly+len_normal))
    elif contamination > anomaly_rate:  # 異常系を減らす
        k = int(np.ceil(len_normal * (anomaly_rate / (1 - anomaly_rate))))
        all = k + len_normal
        print("i : " + str(i) + "\tk : " + str(k) + "\ta\t" + str(all*anomaly_rate))

    else:  # 正常系をへらす
        k = int(len_anomaly / anomaly_rate - len_anomaly)
        all = k + len_anomaly
        print("i : " + str(i) + "\tk : " + str(k) + "\tb\t" + str(all*(1-anomaly_rate)))
