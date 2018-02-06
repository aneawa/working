import sys
import math
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

x = np.random.normal(50, 10, 1000)
X = np.arange(0,100,0.1)
Y = norm.pdf(X,30,10)
YY = norm.pdf(X,60,20)

b = np.arange(0,100,0.1)[0:381]
b2 = np.arange(0,100,0.1)[381:]

Y1 = norm.pdf(X,30,10)[0:381]
YY1 = norm.pdf(X,60,20)[0:381]


plt.plot(X,Y,color='r')
plt.plot(X,YY,color='b')
plt.tick_params(labelbottom="off",bottom="off") # x軸の削除
plt.tick_params(labelleft="off",left="off") # y軸の削除
plt.fill_between(b,YY1,0,facecolor='k',alpha=0.5)
plt.fill_between(b,Y1,0,facecolor='m',alpha=0.5)

plt.plot([38, 38], [0, 0.04], 'k', ls=':')
plt.plot([0, 0], [0, 0.04], 'k', ls=':')
plt.plot([100, 100], [0, 0.04], 'k', ls=':')


plt.xlabel("data")
plt.ylabel("Likelihood")
plt.show()