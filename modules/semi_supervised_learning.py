# Created by RitsukiShuto on 2022/11/23.
# supervised_learning.py
# 半教師あり学習を行う
#
import pandas as pd
import numpy as np
import datetime
import os

from sklearn.model_selection import train_test_split

from modules.model import model_fit
from modules.utils import calc_score, calc_conf_mat, save_fig

def semi_supervised_learning(X_train, Y_train, Z_train, X_test, Y_test, Z_test):
    # ログを保存
    # 現在時刻を文字列として格納
    now = datetime.datetime.now()       # 現在時刻を取得
    time = now.strftime('%Y%m%d_%H%M')
    save_dir = "train_log/semi_supervised/" + time
    os.mkdir(save_dir)

    # ラベルなしデータを読み込む
    un_labeled_U = pd.read_csv("train_data/mixed/POW_un_labeled.csv", header=0, index_col=0)
    un_labeled_V = pd.read_csv("train_data/mixed/TF-IDF_un_labeled.csv", header=0, index_col=0)
    un_labeled_U = un_labeled_U.to_numpy()
    un_labeled_V = un_labeled_V.to_numpy()

    X_train, U_train, Y_train, V_train, Z_train, W_train = train_test_split(X_train, Y_train, Z_train, shuffle=True, test_size=0.5, random_state=0, stratify=Z_train)

    # ラベルなしデータと結合
    U_train = np.append(U_train, un_labeled_U, axis=0)
    V_train = np.append(V_train, un_labeled_V, axis=0)

    print(f"\n教師ありデータ:{X_train.shape[0]}\n教師なしデータ:{U_train.shape[0]}\nテストデータ:{X_test.shape[0]}\n")  # type: ignore

    # ループ回数等に関わる変数
    data_cnt = U_train.shape[0]   # データ件数
    ref_dara_range = 200
    loop_times = data_cnt / ref_dara_range      # ループ回数

    data_count_to_add = -30

    # 実験回数
    experiment_times = 1
    batch_size = 64

    conf_mats = np.zeros((experiment_times, 5, 5))

    for i in range(experiment_times):
        print(f"実験回数:{i+1}/{experiment_times}")

        start = 0       # 未ラベルデータの参照範囲
        end = ref_dara_range

        epochs = 250

        # 初回学習
        multimodal_model, X_single_model, Y_single_model, model_MM, model_X, model_Y = model_fit(X_train, Y_train, Z_train, epochs)
        calc_score(multimodal_model, X_single_model, Y_single_model, X_test, Y_test, Z_test)        # 精度を表示

        # ラベルなしデータを推定して仮ラベルを付与する
        for j in range(int(loop_times) + 1):
            print(f"{j+1}/{int(loop_times)}")
            print(start, "to", end)

            # ラベルなしデータを推定
            MM_encoded = multimodal_model.predict(x=[U_train[start:end][0:], V_train[start:end][0:]], batch_size=batch_size)

            # 信頼度が高い順に20件のデータをピックアップ
            topN_index = np.argpartition(np.max(MM_encoded, axis=1), data_count_to_add)[data_count_to_add:]
            temp_label = np.zeros((-data_count_to_add, 5), dtype=int)

            # 仮ラベルの信頼度と実際のラベルを表示
            correct = 0
            
            for k in range(len(topN_index)):
                # ラベルなしデータかどうかを判別
                if len(W_train) <= topN_index[k] + start:
                    print(f"{topN_index[k] + start}\t{np.argmax(MM_encoded[topN_index[k]])}\t{np.max(MM_encoded[topN_index[k]])}\tNULL")

                else:
                    print(f"{topN_index[k] + start}\t{np.argmax(MM_encoded[topN_index[k]])}\t{np.max(MM_encoded[topN_index[k]])}\t{np.argmax(W_train[topN_index[k] + start])}")

                    # 推定ラベルのうち正解だったものをカウント
                    if np.argmax(MM_encoded[topN_index[k]]) == np.argmax(W_train[topN_index[k] + start]):
                        correct += 1

            print(f"仮ラベル正解率: {correct / -(data_count_to_add)}")

            # 仮ラベル付け
            for l in range(len(topN_index)):
                temp_label[l][np.argmax(MM_encoded[topN_index[l]])] = 1

            X_train = np.append(X_train, U_train[topN_index + start], axis=0)      # 教師ありデータにスタック
            Y_train = np.append(Y_train, V_train[topN_index + start], axis=0)
            Z_train = np.append(Z_train, temp_label,axis=0)

            # データをシャッフル
            np.random.seed(0)              # ランダムシードを固定
            np.random.shuffle(X_train)      # シャッフル

            np.random.seed(0)
            np.random.shuffle(Y_train)

            np.random.seed(0)
            np.random.shuffle(Z_train)

            start = end + 1
            if i == int(loop_times):
                end = data_cnt - 1      # 最後のループのときは[データ長-1]の値をendにする
            else:
                end += ref_dara_range

            epochs += 50

            multimodal_model, X_single_model, Y_single_model, history_MM, model_X, model_Y = model_fit(X_train, Y_train, Z_train, epochs)
            calc_score(multimodal_model, X_single_model, Y_single_model, X_test, Y_test, Z_test)        # 精度を表示

            # lossとaccuracyのグラフを保存
            save_fig(save_dir, multimodal_model, history_MM, i+1, j+1)

        MM_conf_mat = calc_score(multimodal_model, X_single_model, Y_single_model, X_test, Y_test, Z_test)
        MM_conf_mat = np.reshape(MM_conf_mat, (1, 5, 5))
        conf_mats[i, :, :] = MM_conf_mat

    calc_conf_mat(conf_mats, experiment_times, save_dir)