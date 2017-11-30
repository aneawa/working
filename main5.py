
#filename : matファイルの絶対パス
#Xtrains_percent : 訓練データの割合
#maxfeature : max_features
#fit_ylabel : trueならX_trains_correct, falseならNone
#nn_estimator : n_estimators ツリー数
#sepaLabel : trueならseparate, falseならmixed
#treeLabel : true : 木の実験用のprint

from test_isolationforest import main
import datetime

hoge = 10

filename = '/home/anegawa/Dropbox/satellite.mat'

# for i in range(1, 401):
#     auc = 0
#     for j in range(hoge):
#         auc2 = main(filename, xtrains_percent = 0.5 , maxfeature = 3, fit_ylabel = False , nn_estimator = i,
#                     sepaLabel = False, treeLabel=True, seed = datetime.datetime.today().microsecond, pcaLabel = False)
#         auc +=auc2
#     print(auc/hoge)

for i in [0.25, 0.2, 0.1]:
    auc = 0
    for l in range(hoge):
        auc2 = main(filename, xtrains_percent=i, fit_ylabel=False, nn_estimator=100,
                    sepaLabel=True, treeLabel=True, seed=datetime.datetime.today().microsecond, pcaLabel=False,
                    n_comp=i, sepa2=False, time_label=False, stream=False, sfl=False, anomaly_rate=0.68)
        auc += auc2
    print(auc / hoge)