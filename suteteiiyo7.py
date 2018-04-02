import sys
import math
import statistics
import numpy as np
from scipy.stats import norm
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

def H(i):
    return math.log(i) + 0.5772156649

def c(n):
    return 2 * H(n-1) - (2*(n-1)/n)


def E(data):
    return statistics.mean(data)

def s(x, n):
    return pow(2, -(x/c(n)))

hoge = []
x = 130
for i in range(x):
    hoge.append(s(i,100))

plt.title("The relation between E(h(x)) and s")
plt.plot(hoge)
plt.xlabel('E(h(x))')
plt.ylabel('s')
plt.xticks([0,x], ["0","n - 1"])
# plt.ylim(0, 0.225)
# plt.xlim(0, 37)
# plt.legend(loc = "upper right")
# plt.grid(True)
plt.show()