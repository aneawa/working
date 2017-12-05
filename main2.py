
#filename : matファイルの絶対パス
#Xtrains_percent : 訓練データの割合
#maxfeature : max_features
#fit_ylabel : trueならX_trains_correct, falseならNone
#nn_estimator : n_estimators ツリー数
#sepaLabel : trueならseparate, falseならmixed
#treeLabel : true : 木の実験用のprint

from test_isolationforest import main
import datetime
import numpy as np

hoge = 10
hoge2 = 36
auc = 0
filename = '/home/anegawa/Dropbox/ionosphere.mat'
#all_result = [all, pca_train(fit+trans), fit, pca_test(trans), test, sum_train, sum_test]

# for i in range(10, 110, 10):
for i in range(0, 101):
    # all_result = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    all_result = np.array([0.0 ,0.0])
    for j in range(hoge):
        result = main(filename, xtrains_percent = 0.2 , maxfeature = 3, fit_ylabel = False, nn_estimator = 100,
                    sepaLabel = True, treeLabel=True, seed = datetime.datetime.today().microsecond, pcaLabel = False,
                    n_comp = 31, sepa2 = False, time_label=False, stream=False, anomaly_rate = i/100)
        all_result += result
    all_result = np.array(all_result)
    print(all_result/hoge)