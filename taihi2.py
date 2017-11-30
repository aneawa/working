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

import scipy.io
import h5py
# from scipy.sparse import csc_matrix, csr_matrix

from decimal import *


def calc_accuracy(X_rabel, X_pred, treeLabel):
    bunbo = len(X_rabel)
    bunshi = 0
    for i in range(bunbo):
        if (X_rabel[i] == X_pred[i]):
            bunshi += 1
    # accuracy = Decimal(bunshi) / Decimal(bunbo)
    accuracy = bunshi / bunbo
    if treeLabel == False:
        print("accuracy = " + repr(accuracy))
    return accuracy


def calc_AUC(label, pred, treeLabel):
    a = np.array(label)
    fpr, tpr, thresholds = roc_curve(label, pred)
    score = auc(fpr, tpr)
    # print("auc2 : "+ str(roc_auc_score(label, pred)))
    if treeLabel:
        # print(score)
        anegawa = 0
    else:
        print("AUC score: " + str(score))
        print("")
    # print("fpr :" + str(fpr))
    # print("tpr :" + str(tpr))
    # print(score)
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr)
    # plt.title("ROC curve")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.show()
    if math.isnan(score):
        majikayo = True
    return score


# sepa2 : train only noraml data
def main(filename, xtrains_percent=0.8, maxfeature=1, fit_ylabel=False, nn_estimator=100, sepaLabel=True,
         treeLabel=False, seed=42, pcaLabel=False, n_comp=2, sepa2=False, time_label=False, stream=False, sfl=False):
    mugen = float("inf")
    all_start = time.time()
    rng = np.random.RandomState(seed)

    # httpとsmtpは別の方法でデータ取得
    if filename == '/home/anegawa/Dropbox/http.mat' or filename == '/home/anegawa/Dropbox/smtp.mat':
        mat = {}
        f = h5py.File(filename)
        for k, v in f.items():
            mat[k] = np.array(v)
        X = mat['X'].T
        y2 = mat['y'][0]
        y3 = []
        for i in range(len(y2)):
            y3.append(int(y2[i]))
        y = np.reshape(y3, [len(y3), 1])


    else:
        mat = scipy.io.loadmat(filename)
        X = mat['X']
        y = mat['y']

    rate = xtrains_percent
    max_feat = int(maxfeature)
    if max_feat == 3:
        max_feat = X.shape[1]

    if treeLabel:
        anegawa = 0
    else:
        print('X_train\'s rate : ' + str(rate))
        print('max_features : ' + str(max_feat))
        print('fit_ylabel : ' + str(fit_ylabel))
        print('nn_estimator : ' + str(nn_estimator))
        print('sepaLabel : ' + str(sepaLabel))

    clf = IsolationForest(random_state=rng)
    clf.n_estimators = nn_estimator
    clf.verbose = 0
    clf.max_features = max_feat

    if (str(filename) == '/home/anegawa/Dropbox/shuttle.mat'):
        clf.contamination = 0.07

    elif (str(filename) == '/home/anegawa/Dropbox/http.mat'):
        clf.contamination = 0.004

    elif (str(filename) == '/home/anegawa/Dropbox/pima.mat'):
        clf.contamination = 0.35

    elif (str(filename) == '/home/anegawa/Dropbox/mammography.mat'):
        clf.contamination = 0.02

    elif (str(filename) == '/home/anegawa/Dropbox/cover.mat'):
        clf.contamination = 0.009

    elif (str(filename) == '/home/anegawa/Dropbox/breastw.mat'):
        clf.contamination = 0.35

    elif (str(filename) == '/home/anegawa/Dropbox/arrhythmia.mat'):
        clf.contamination = 0.15

    elif (str(filename) == '/home/anegawa/Dropbox/ionosphere.mat'):
        clf.contamination = 0.36

    elif (str(filename) == '/home/anegawa/Dropbox/satellite.mat'):
        clf.contamination = 0.32

    elif (str(filename) == '/home/anegawa/Dropbox/annthyroid.mat'):
        clf.contamination = 0.07

    elif (str(filename) == '/home/anegawa/Dropbox/smtp.mat'):
        clf.contamination = 0.03 / 100

    else:
        print('cannot file it.')
        # Generate train data
        a = rng.randn(400, 2)
        X = 0.3 * a
        X_train = np.r_[X + 2, X - 2]
        # X_train = np.ones([400, 2])

        # Generate some regular novel observations
        X = 0.3 * rng.randn(400, 2)
        X_test = np.r_[X + 2, X - 2]
        # X_test = np.ones([400, 2])

        # Generate some abnormal novel observations
        X_outliers = np.random.exponential(1. / 0.001, size=[20, 2])
        # X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
        # X_outliers = np.zeros([20, 2])
        X_test = np.r_[X_test, X_outliers]
        X_train_correct = np.ones([X_train.shape])

    hoge = 1 / (1 - rate)
    cross_count = int(np.ceil(hoge))
    if cross_count > hoge:
        cross_count = cross_count - 1

    sum_auc = 0
    sum_accuracy = 0

    pca_fit_time = 0
    pca_transform_train_time = 0
    pca_transform_test_time = 0
    test_time = 0
    fit_time = 0
    sum_train_time = 0

    # for count in range(cross_count):

    if sepaLabel == True:  # separated
        # data cut
        X_anomaly = []
        X_normal = []
        for i in range(len(X)):
            if y[i] == 1:
                X_anomaly.append(X[i])
            else:
                X_normal.append(X[i])

        cutter_anomaly = int(np.ceil(len(X_anomaly) * rate))
        cutter_normal = int(np.ceil(len(X_normal) * rate))

        for count in range(cross_count):
            part_anomaly = int(np.ceil(cutter_anomaly * count))
            part_normal = int(np.ceil(cutter_normal * count))
            X_train = []
            X_train_correct = []
            X_test = []
            X_test_correct = []

            for i, k in zip(range(len(X_anomaly)), range(part_anomaly, part_anomaly + len(X_anomaly))):
                while k >= len(X_anomaly):
                    k = k - len(X_anomaly)

                if i < cutter_anomaly:
                    X_train.append(X_anomaly[k])
                    X_train_correct.append(-1)

                else:
                    X_test.append(X_anomaly[k])
                    X_test_correct.append(-1)

            for i, k in zip(range(len(X_normal)), range(part_normal, part_normal + len(X_normal))):
                while k >= len(X_normal):
                    k = k - len(X_normal)

                if i < cutter_normal:
                    X_train.append(X_normal[k])
                    X_train_correct.append(1)
                else:
                    X_test.append(X_normal[k])
                    X_test_correct.append(1)

            if sfl:
                X_train_set = []
                X_test_set = []
                for i in range(len(X_train)):
                    buf = []
                    buf.append(X_train[i])
                    buf.append(X_train_correct[i])
                    X_train_set.append(buf)
                for i in range(len(X_test)):
                    buf = []
                    buf.append(X_test[i])
                    buf.append(X_test_correct[i])
                    X_test_set.append(buf)

                random.shuffle(X_train_set)
                random.shuffle(X_test_set)

                X_train = []
                X_test = []
                X_train_correct = []
                X_test_correct = []
                for i in range(len(X_train_set)):
                    X_train.append(X_train_set[i][0])
                    X_train_correct.append(X_train_set[i][1])
                for i in range(len(X_test_set)):
                    X_test.append(X_test_set[i][0])
                    X_test_correct.append(X_test_set[i][1])





    else:  # mixed
        cutter = len(X) * rate  # test start this index at the first time
        for count in range(cross_count):
            part = int(np.ceil(cutter * count))
            # while part >= len(X):
            #     part = part - len(X)
            X_train = []
            X_train_correct = []
            X_test = []
            X_test_correct = []

            for i, k in zip(range(len(X)), range(part, part + len(X))):
                while k >= len(X):
                    k = k - len(X)

                if i < len(X) * rate:
                    X_train.append(X[k])
                    X_train_correct.append(y[k])

                else:
                    X_test.append(X[k])
                    X_test_correct.append(y[k])

            for q in range(len(X_train_correct)):
                j = X_train_correct[q]
                if (j == 1):
                    X_train_correct[q] = -1
                else:
                    X_train_correct[q] = 1

            for w in range(len(X_test_correct)):
                j = X_test_correct[w]
                if (j == 1):
                    X_test_correct[w] = -1
                else:
                    X_test_correct[w] = 1

        # owari
        # finished cutting data

        if pcaLabel:
            pca_fit_start = time.time()
            pca = PCA(copy=True, iterated_power='auto', n_components=n_comp, random_state=None,
                      svd_solver='auto', tol=0.0, whiten=False)
            pca2 = PCA(copy=True, iterated_power='auto', random_state=None,
                       svd_solver='auto', tol=0.0, whiten=False)

            if sepa2:
                # if False:
                print("こっち入ってるけどええんか!?")
                pca2.fit(X_train_normal)
                component = pca2.components_
                component2 = np.sort(pca2.components_)
                if n_comp < len(component2):
                    pca2.components_ = component2[0:n_comp]
                    # print(pca2.components_.shape)
                X_train = pca2.transform(X_train)
                X_test = pca2.transform(X_test)

            else:
                pca.fit(X_train)
                pca_fit_finish = time.time()

                pca_transform_train_start = time.time()
                X_train = pca.transform(X_train)
                pca_transform_train_finish = time.time()

                # a = X_test[0]
                # X_test = pca.transform(a)

                # if not stream:
                #     pca_transform_test_start = time.time()
                #     X_test = pca.transform(X_test) #stream version
                #     pca_transform_test_finish = time.time()
                #     pca_transform_test_time += (pca_transform_test_finish - pca_transform_test_start)
            clf.max_features = n_comp
            pca_fit_time += (pca_fit_finish - pca_fit_start)
            pca_transform_train_time += (pca_transform_train_finish - pca_transform_train_start)

        fit_start = time.time()
        if fit_ylabel:
            clf.fit(X_train, X_train_correct, sample_weight=None)
        else:
            clf.fit(X_train, y=None, sample_weight=None)
        fit_finish = time.time()
        fit_time += (fit_finish - fit_start)

        # if pcaLabel and stream:
        if stream:
            sum_score_auc = []
            sum_score_acc = []

            # print(X_test[0:1])
            for i in range(len(X_test)):
                if pcaLabel:
                    pca_transform_test_start = time.time()
                    a = [X_test[i]]
                    X_test_pca = pca.transform(a)
                    pca_transform_test_finish = time.time()
                    pca_transform_test_time += (pca_transform_test_finish - pca_transform_test_start)

                else:
                    X_test_pca = [X_test[i]]

                test_start = time.time()
                y_pred_test, a_score = clf.predict(X_test_pca)
                test_finish = time.time()
                test_time += (test_finish - test_start)

                sum_score_auc.append(a_score)
                sum_score_acc.append(y_pred_test)
            a_score = sum_score_auc
            y_pred_test = sum_score_acc

        else:
            if pcaLabel:
                pca_transform_test_start = time.time()
                X_test = pca.transform(X_test)  # stream version
                pca_transform_test_finish = time.time()
                pca_transform_test_time += (pca_transform_test_finish - pca_transform_test_start)

            test_start = time.time()
            y_pred_test, a_score = clf.predict(X_test)
            test_finish = time.time()
            test_time += (test_finish - test_start)
        # a_score = clf.decision_function(X_test)

        acc = calc_accuracy(X_test_correct, y_pred_test, treeLabel)
        AUC = calc_AUC(X_test_correct, a_score, treeLabel)
        sum_auc += AUC
        sum_accuracy = acc

    # return AUC





    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # plot the line, the samples, and the nearest vectors to the plane
    # xx, yy = np.meshgrid(np.linspace(-200, 200, 1000), np.linspace(-200, 200, 1000))
    # # clf.max_features = 2
    # # print(yy.ravel())
    # # Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    #
    # # Z = Z.reshape(xx.shape)
    #
    # plt.figure(figsize=(100, 200))
    # plt.suptitle("satellite")
    # # plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    #
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    #
    # lim = True
    # x = (-200, 200)
    # y = (-200, 300)
    #
    # for i,j in zip(range(2), [True, False]):
    #     small = j  # trueがsmallestね
    #
    #     plt.subplot(2, 2, i+1)
    #     if small:
    #         plt.title("smallest")
    #     else:
    #         plt.title("largest")
    #
    #     if small:
    #         # b1 = plt.scat
    # # plot the line, the samples, and the nearest vectors to the plane
    # xx, yy = np.meshgrid(np.linspace(-200, 200, 1000), np.linspace(-200, 200, 1000))
    # # clf.max_features = 2
    # # print(yy.ravel())
    # # Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    #
    # # Z = Z.reshape(xx.shape)
    #
    # plt.figure(figsize=(100, 200))
    # plt.suptitle("satellite")
    # # plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    #
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    #
    # lim = True
    # x = (-200, 200)
    # y = (-200, 300)
    #
    # for i,j in zip(range(2), [True, False]):
    #     small = j  # trueがsmallestね
    #
    #     plt.subplot(2, 2, i+1)
    #     if small:
    #         plt.title("smallest")
    #     else:
    #         plt.title("largest")
    #
    #     if small:
    #         # b1 = plt.scatter(X_train[:, X_train.shape[1]-1], X_train[:, X_train.shape[1]-2], c='white', s=20, edgecolor='k')
    #         b2 = plt.scatter(X_test[:, X_test.shape[1]-1], X_test[:, X_test.shape[1]-2], c='green', s=20, edgecolor='k')
    #     else:
    #         # b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k')
    #         b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green', s=20, edgecolor='k')
    #     # c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')
    #     plt.axis('tight')
    #     if lim:
    #         plt.xlim(x)
    #         plt.ylim(y)
    #     # plt.legend([b1, b2],
    #     #            ["training observations",
    #     #             "testing observations"],
    #     #            loc="upper left")
    #     plt.legend([b2],["testing observations"],
    #                loc="upper left")
    #     # plt.legend([b1], ["training observations"],
    #     #            loc="upper left")
    #
    #
    #
    #     plt.subplot(2, 2, i+3)
    #     if small:
    #         b1 = plt.scatter(X_train[:, X_train.shape[1]-1], X_train[:, X_train.shape[1]-2], c='white', s=20, edgecolor='k')
    #         # b2 = plt.scatter(X_test[:, X_test.shape[1] - 1], X_test[:, X_test.shape[1] - 2], c='green', s=20, edgecolor='k')
    #     else:
    #         b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k')
    #         # b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green', s=20, edgecolor='k')
    #     # c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')
    #     plt.axis('tight')
    #     if lim:
    #         plt.xlim(x)
    #         plt.ylim(y)
    #     # plt.legend([b1, b2],
    #     #            ["training observations",
    #     #             "testing observations"],
    #     #            loc="upper left")
    #     # plt.legend([b2], ["testing observations"],
    #     #            loc="upper left")
    #     plt.legend([b1], ["training observations"],
    #                loc="upper left")
    # plt.show()ter(X_train[:, X_train.shape[1]-1], X_train[:, X_train.shape[1]-2], c='white', s=20, edgecolor='k')
    #         b2 = plt.scatter(X_test[:, X_test.shape[1]-1], X_test[:, X_test.shape[1]-2], c='green', s=20, edgecolor='k')
    #     else:
    #         # b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k')
    #         b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green', s=20, edgecolor='k')
    #     # c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')
    #     plt.axis('tight')
    #     if lim:
    #         plt.xlim(x)
    #         plt.ylim(y)
    #     # plt.legend([b1, b2],
    #     #            ["training observations",
    #     #             "testing observations"],
    #     #            loc="upper left")
    #     plt.legend([b2],["testing observations"],
    #                loc="upper left")
    #     # plt.legend([b1], ["training observations"],
    #     #            loc="upper left")
    #
    #
    #
    #     plt.subplot(2, 2, i+3)
    #     if small:
    #         b1 = plt.scatter(X_train[:, X_train.shape[1]-1], X_train[:, X_train.shape[1]-2], c='white', s=20, edgecolor='k')
    #         # b2 = plt.scatter(X_test[:, X_test.shape[1] - 1], X_test[:, X_test.shape[1] - 2], c='green', s=20, edgecolor='k')
    #     else:
    #         b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k')
    #         # b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green', s=20, edgecolor='k')
    #     # c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')
    #     plt.axis('tight')
    #     if lim:
    #         plt.xlim(x)
    #         plt.ylim(y)
    #     # plt.legend([b1, b2],
    #     #            ["training observations",
    #     #             "testing observations"],
    #     #            loc="upper left")
    #     # plt.legend([b2], ["testing observations"],
    #     #            loc="upper left")
    #     plt.legend([b1], ["training observations"],
    #                loc="upper left")
    # plt.show()
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




    auc2 = sum_auc / cross_count
    # print(sum_accuracy)
    acc2 = sum_accuracy / cross_count

    # calc time
    all_finish = time.time()
    all_time = all_finish - all_start
    pca_fit_time = pca_fit_time / cross_count
    pca_transform_train_time = pca_transform_train_time / cross_count
    pca_transform_test_time = pca_transform_test_time / cross_count
    test_time = test_time / cross_count
    fit_time = fit_time / cross_count
    sum_train_time = fit_time + pca_fit_time + pca_transform_train_time
    sum_test_time = pca_transform_test_time + test_time
    # print("sum_train_time : " + str(sum_train_time))
    # print("pca_transform_train_time : " + str(pca_transform_train_time))
    # print("pca_fit_time : " + str(pca_fit_time))
    # print("test_time : " + str(test_time))
    # print("fit_time : " + str(fit_time))
    # print("all_time : " + str(all_time))
    # return

    if time_label:
        return all_time, pca_fit_time + pca_transform_train_time, fit_time, pca_transform_test_time, test_time, sum_train_time, sum_test_time
    elif treeLabel:
        if math.isnan(auc2):
            majikayo = True
        return auc2
    else:
        return auc2, acc2
