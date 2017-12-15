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

from decimal import *

#歪度(skewness), 尖度(kurtosis)を計算
def kurtosis_skewness(data):
    data = np.array(data)
    ave = np.average(data)
    std = np.std(data)
    kurtosis = np.average((data - ave)**3) / (std**3)
    skewness = np.average((data - ave)**4) / (std**4) - 3

    return kurtosis, skewness


def calc_accuracy(X_rabel, X_pred, treeLabel): #精度計算(Accuracy)
    bunbo = len(X_rabel)
    bunshi = 0
    for i in range(bunbo):
        if (X_rabel[i] == X_pred[i]):
            bunshi += 1
    # accuracy = Decimal(bunshi) / Decimal(bunbo)
    accuracy = bunshi / bunbo
    if not treeLabel:
        print("accuracy = " + repr(accuracy))
    return accuracy


def calc_AUC(label, pred, treeLabel): #精度計算(AUC)
    roc = True
    if roc:
        fpr, tpr, thresholds = roc_curve(label, pred)
        score_roc = auc(fpr, tpr)
        # score_roc = roc_auc_score(label, pred)
        # #------------------------------
        # print("fpr :" + str(fpr))
        # print("tpr :" + str(tpr))
        # print(score_roc)
        # plt.figure(figsize=(8, 6))
        # plt.plot(fpr, tpr)
        # plt.title("ROC curve")
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.show()

        # if math.isnan(score_roc):
        #     raise Exception("error! auc is NaN!")

    else:
        (precision, recall, _) = metrics.precision_recall_curve(label, pred)
        score_pr = auc(recall, precision)

    if not treeLabel:
        print("AUC score_roc: " + str(score_roc))
        print("AUC score_pr: " + str(score_pr))
        # print("auc2 : " + str(score_roc))

    if roc:
        return score_roc
    else:
        return score_pr

def calc_FN(label, pred):
    n_data = len(label)
    FP = 0
    TP = 0
    TN = 0
    FN = 0
    for i in range(len(label)):
        if label[i] == -1:
            if pred[i] == -1:
                TP += 1
            else:
                FN += 1
        else:
            if pred[i] == -1:
                FP += 1
            else:
                TN += 1
    FPR = FP / n_data
    TPR = TP / n_data
    TNR = TN / n_data
    FNR = FN / n_data

    return FPR, TPR, TNR, FNR



#filename        : ファイル名(データ名) string
#xtrains_percent : 訓練データの割合 float
#fit_ylabel      : fit時のyをlabelにするかNoneにするか(以前使っていましたが今はFalse固定) bool
#nn_estimator    : ツリー数 int
#sepaLabel       : データを学習用と評価用に分ける際，正常系と異常系それぞれからxtrains_percent分取るかどうか bool
#treeLabel       : 出力する仕様の変更(TrueだとAUCのみを出力, Falseなら他の情報も出力) bool
#seed            : 乱数のシード値 int
#pcaLabel        : pcaを行うか否か bool
#n_comp          : pcaLabel == Trueのときの取得するコンポーネント数 int
#sepa2           : 通常系の分散を小さくするように軸を取る時に使用した(今はFalse固定) bool
#time_label      : aucでなく実行時間を出力(treeLabelより優先) bool
#stream          : リアルタイム処理 bool
#sfl             : sepaLabel == Trueのときデータをシャッフルするか否か bool

def main(filename, xtrains_percent = 0.8, maxfeature = None, fit_ylabel = False, nn_estimator = 100, sepaLabel = True,
         treeLabel = False, seed = 42, pcaLabel = False, n_comp = 2, sepa2 = False, time_label = False, stream = False,
         sfl = False, anomaly_rate = None, max_samples = None):

    inf = float("inf")
    all_start = time.time()
    rng = np.random.RandomState(seed)

    #httpとsmtpのみ別の方法でデータ取得
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

    if maxfeature == None:
        max_feat = X.shape[1]
    else:
        max_feat = int(maxfeature)

    if not treeLabel:
        print('X_train\'s rate : ' + str(rate))
        print('max_features : ' + str(max_feat))
        print('fit_ylabel : ' + str(fit_ylabel))
        print('nn_estimator : ' + str(nn_estimator))
        print('sepaLabel : ' + str(sepaLabel))

    clf = IsolationForest(random_state=rng)
    clf.n_estimators = nn_estimator
    clf.verbose = 0
    clf.max_features = max_feat
    if max_samples != None:
        clf.max_samples = max_samples

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
        raise Exception("error! cannot file it.")

    #交差検証を何回行うか(例:8:2なら5回)
    #もっとうまい方法ありそう
    hoge = 1 / rate
    cross_count = int(np.ceil(hoge))
    if cross_count > hoge:
        cross_count = cross_count - 1


    #cross_count分のauc,acc合計
    sum_auc_roc = 0
    sum_accuracy = 0
    FPR_sum = 0
    TPR_sum = 0
    TNR_sum = 0
    FNR_sum = 0


    pca_fit_time = 0
    pca_transform_train_time = 0
    pca_transform_test_time = 0
    test_time = 0
    fit_time = 0

    #ここはデータを交差検証用に分割するだけ
    if sepaLabel == True:  # separated
        # data cut
        X_anomaly = []
        X_normal = []
        for i in range(len(X)):
            if y[i] == 1:
                X_anomaly.append(X[i])
            else:
                X_normal.append(X[i])

        #データ全体のcontaminationを操作
        zentai = False
        if zentai:
            if anomaly_rate != clf.contamination:
                clf.contamination = anomaly_rate
                if anomaly_rate < clf.contamination:
                    #異常系をカットしますよ〜〜
                    k = int(np.ceil(len(X_normal) * (anomaly_rate / (1 - anomaly_rate))))
                    anomaly_hoge = random.sample(X_anomaly, k)  # ランダムに抽出
                    normal_hoge = X_normal
                else:
                    #正常系をカットしましょうね〜〜
                    n_normal = int(len(X_anomaly) / anomaly_rate) - len(X_anomaly)
                    normal_rate = n_normal / len(X_normal)
                    k = int(np.ceil(len(X_normal) * normal_rate))
                    normal_hoge = random.sample(X_normal, k)  # ランダムに抽出
                    anomaly_hoge = X_anomaly
                X_anomaly = anomaly_hoge
                X_normal = normal_hoge
            anomaly_rate = None


        cutter_anomaly = len(X_anomaly) * rate
        cutter_normal = len(X_normal) * rate
        X_sepa_ano = []
        X_sepa_nor = []
        for i in range(cross_count):
            head2 = int(cutter_normal * i)
            tail2 = int(cutter_normal * (i+1)) - 1
            X_sepa_nor.append(X_normal[head2:tail2+1])

            head = int(cutter_anomaly * i)
            tail = int(cutter_anomaly * (i+1)) - 1
            X_sepa_ano.append(X_anomaly[head:tail+1])



    else:
        X_sepa = []
        y_sepa = []
        cutter = len(X)*rate
        for i in range(cross_count):
            head = int(cutter*i)
            tail = int(cutter*(i+1))-1
            X_sepa.append(X[head:tail+1])
            y_sepa.append(y[head:tail+1])




    for count in range(cross_count):
        if sepaLabel:
            X_train = []
            X_train_correct = []
            X_test = []
            X_test_correct = []

            # 学習データの異常系含有率を変更
            for i in range(cross_count):
                if i != count:
                    train_flag = True
                    if anomaly_rate is not None:
                        if clf.contamination != anomaly_rate:
                            train_flag = False
                            if clf.contamination > anomaly_rate:  # 異常系を減らす
                                k = int(np.ceil(len(X_sepa_nor[i]) * (anomaly_rate / (1 - anomaly_rate))))
                                anomaly_hoge = random.sample(X_sepa_ano[i], k)  # ランダムに抽出
                                normal_hoge = X_sepa_nor[i]
                                # X_sepa_ano[i] = anomaly_hoge
                                # X_sepa_ano[i] = []
                                # X_sepa_ano[i].extend(anomaly_hoge)
                            else:  # 正常系を減らす
                                n_normal = int(len(X_sepa_ano[i]) / anomaly_rate) - len(X_sepa_ano[i])
                                normal_rate = n_normal / len(X_sepa_nor[i])
                                k = int(np.ceil(len(X_sepa_nor[i]) * normal_rate))
                                normal_hoge = random.sample(X_sepa_nor[i], k)  # ランダムに抽出
                                anomaly_hoge = X_sepa_ano[i]
                                # X_sepa_nor[i] = normal_hoge
                            X_train.extend(anomaly_hoge)
                            for j in range(len(anomaly_hoge)):
                                X_train_correct.append(-1)

                            X_train.extend(normal_hoge)
                            for j in range(len(normal_hoge)):
                                X_train_correct.append(1)

                    if train_flag:
                        X_train.extend(X_sepa_ano[i])
                        for j in range(len(X_sepa_ano[i])):
                            X_train_correct.append(-1)

                        X_train.extend(X_sepa_nor[i])
                        for j in range(len(X_sepa_nor[i])):
                            X_train_correct.append(1)

                else:
                    # X_test.extend(X_sepa_ano[i])
                    # for j in range(len(X_sepa_ano[i])):
                    #     X_test_correct.append(-1)
                    # X_test.extend(X_sepa_nor[i])
                    # for j in range(len(X_sepa_nor[i])):
                    #     X_test_correct.append(1)

                    #テストデータの含有率も変えます？？
                    anomaly_rate2 = None
                    test_flag = True
                    if anomaly_rate2 is not None:
                        clf.contamination = anomaly_rate2
                        if clf.contamination != anomaly_rate2:
                            test_flag = False
                            if clf.contamination > anomaly_rate2:  # 異常系を減らす
                                k = int(np.ceil(len(X_sepa_nor[i]) * (anomaly_rate / (1 - anomaly_rate))))
                                anomaly_hoge = random.sample(X_sepa_ano[i], k)  # ランダムに抽出
                                normal_hoge = X_sepa_nor[i]
                                # X_sepa_ano[i] = anomaly_hoge
                                # X_sepa_ano[i] = []
                                # X_sepa_ano[i].extend(anomaly_hoge)
                            else:  # 正常系を減らす
                                n_normal = int(len(X_sepa_ano[i]) / anomaly_rate) - len(X_sepa_ano[i])
                                normal_rate = n_normal / len(X_sepa_nor[i])
                                k = int(np.ceil(len(X_sepa_nor[i]) * normal_rate))
                                normal_hoge = random.sample(X_sepa_nor[i], k)  # ランダムに抽出
                                anomaly_hoge = X_sepa_ano[i]
                                # X_sepa_nor[i] = normal_hoge
                            X_test.extend(anomaly_hoge)
                            for j in range(len(anomaly_hoge)):
                                X_test_correct.append(-1)

                            X_test.extend(normal_hoge)
                            for j in range(len(normal_hoge)):
                                X_test_correct.append(1)

                    if test_flag:
                        X_test.extend(X_sepa_ano[i])
                        for j in range(len(X_sepa_ano[i])):
                            X_test_correct.append(-1)

                        X_test.extend(X_sepa_nor[i])
                        for j in range(len(X_sepa_nor[i])):
                            X_test_correct.append(1)



            #シャッフルするかどうか
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





        else: #mixed
            X_train = []
            X_train_correct = []
            X_test = []
            X_test_correct = []
            for i in range(cross_count):
                if i != count:
                    X_train.extend(X_sepa[i])
                    X_train_correct.extend(y_sepa[i])
                else:#i == count
                    print(i, 1111111)
                    X_test.extend(X_sepa[i])
                    X_test_correct.extend(y_sepa[i])

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
            pca.fit(X_train)
            pca_fit_finish = time.time()

            pca_transform_train_start = time.time()
            X_train = pca.transform(X_train)
            pca_transform_train_finish = time.time()

            clf.max_features = n_comp
            pca_fit_time += (pca_fit_finish - pca_fit_start)
            pca_transform_train_time += (pca_transform_train_finish - pca_transform_train_start)


        varLabel = False
        kurtosisLabel = True
        if varLabel or kurtosisLabel:
            var = []
            kurtosis_set = []
            skewness_set = []
            for i in range(len(X_train[0])):
                data = []
                for j in range(len(X_train)):
                    data.append(X_train[j][i])
                data = np.array(data)
                ave = np.average(data)
                std = np.std(data)
                kurtosis = np.average((data - ave) ** 3) / (std ** 3)
                skewness = np.average((data - ave) ** 4) / (std ** 4) - 3

                var.append(std)
                kurtosis_set.append(kurtosis)
                skewness_set.append(skewness)
            var_rank = np.argsort(var)[::-1]
            kurtosis_rank = np.argsort(kurtosis_set)[::-1]
            skewness_rank = np.argsort(skewness_set)[::-1] #歪度は使う予定今のとこないかな〜

            hoge = []
            for i in range(clf.max_features):
                if varLabel:
                    hoge.append(np.array(X_train)[:,var_rank[i]])
                elif kurtosisLabel:
                    hoge.append(np.array(X_train)[:,kurtosis_rank[i]])
            print(X_train)
            X_train = np.array(hoge).T
            print(X_train)
            print("")

        fit_start = time.time()
        #fit_ylabelはFalseで固定
        if fit_ylabel:
            clf.fit(X_train, X_train_correct, sample_weight=None)
        else :
            clf.fit(X_train, y = None, sample_weight=None)
        fit_finish = time.time()
        fit_time += (fit_finish - fit_start)


        if stream:
            sum_score_auc = []
            sum_score_acc = []

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

        else: #batch
            if pcaLabel:
                pca_transform_test_start = time.time()
                X_test = pca.transform(X_test)  # stream version
                pca_transform_test_finish = time.time()
                pca_transform_test_time += (pca_transform_test_finish - pca_transform_test_start)

            test_start = time.time()
            y_pred_test, a_score = clf.predict(X_test)
            # a_score = clf.decision_function(X_test)
            test_finish = time.time()
            test_time += (test_finish - test_start)


        acc = calc_accuracy(X_test_correct, y_pred_test, treeLabel)
        AUC_roc = calc_AUC(X_test_correct, a_score, treeLabel)
        FPR, TPR, TNR, FNR = calc_FN(X_test_correct, y_pred_test)
        FNR_sum += FNR
        FPR_sum += FPR
        sum_auc_roc += AUC_roc
        sum_accuracy += acc



    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # plot the line, the samples, and the nearest vectors to the plane
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




    auc2_roc = sum_auc_roc / cross_count
    acc2 = sum_accuracy / cross_count
    fnr = FNR_sum / cross_count
    fpr = FPR_sum / cross_count

    #calc time
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

    if time_label:
        return all_time, pca_fit_time + pca_transform_train_time, fit_time, pca_transform_test_time, test_time, sum_train_time, sum_test_time
    elif treeLabel:
        # if math.isnan(auc2_roc):
        #     raise Exception("error! auc is NaN!.")
        return auc2_roc
        # return fnr
        # return fpr

    else:
        return auc2_roc, acc2
