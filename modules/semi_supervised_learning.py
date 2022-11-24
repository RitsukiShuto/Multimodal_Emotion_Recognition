# Created by RitsukiShuto on 2022/11/23.
# supervised_learning.py
# 半教師あり学習を行う
#
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from modules.model import model_fit
from modules.evaluate_model import calc_score

def semi_supervised_learning(X_train, Y_train, Z_train, X_test, Y_test, Z_test):
    # ラベルなしデータを読み込む
    un_labeled_U = pd.read_csv("train_data/mixed/MFCC_un_labeled.csv", header=0, index_col=0)
    un_labeled_V = pd.read_csv("train_data/mixed/TF-IDF_un_labeled.csv", header=0, index_col=0)

    # データを変換
    un_labeled_U = un_labeled_U.to_numpy()
    un_labeled_V = un_labeled_V.to_numpy()

    X_train, U_train, Y_train, V_train, Z_train, W_train = train_test_split(
        X_train, Y_train, Z_train, shuffle=True, test_size=0.8, random_state=0, stratify=Z_train)

    # ラベルなしデータと結合
    U_train = np.append(U_train, un_labeled_U, axis=0)
    V_train = np.append(V_train, un_labeled_V, axis=0)

    print(f"教師ありデータ:{X_train.shape[0]}\n教師なしデータ:{U_train.shape[0]}\nテストデータ:{X_test.shape[0]}")

    # ループ回数等に関わる変数
    data_cnt = U_train.shape[0]   # データ件数
    ref_dara_range = 100
    loop_times = data_cnt / ref_dara_range      # ループ回数
    last_loop = data_cnt - ref_dara_range       # TODO: ラベルなしデータの端数部を処理するための変数

    batch_size = 64

    for i in range(10):
        print(f"{i}/10")

        start = 0       # 未ラベルデータの参照範囲
        end = ref_dara_range

        epochs = 250
        conf_mats = []

        # 初回学習
        multimodal_model, X_single_model, Y_single_model, model_MM, model_X, model_Y = model_fit(X_train, Y_train, Z_train, epochs)

        for j in range(int(loop_times)):
            print(j+1, "/", int(loop_times))
            print(start, "to", end)

            # ラベルなしデータを推定
            MM_encoded = multimodal_model.predict(x=[U_train[start:end][0:], V_train[start:end][0:]], batch_size=batch_size)

            # 信頼度が高い順に20件のデータをピックアップ
            top20_index = np.argpartition(np.max(MM_encoded, axis=1), -20)[-20:]
            temp_label = np.zeros((20, 5), dtype=int)

            # 仮ラベル付け
            for l in range(len(top20_index)):
                temp_label[l][np.argmax(MM_encoded[top20_index[l]])] = 1

            X_train = np.append(X_train, U_train[top20_index + start], axis=0)      # 教師ありデータにスタック
            Y_train = np.append(Y_train, V_train[top20_index + start], axis=0)
            Z_train = np.append(Z_train, temp_label,axis=0)

            # データをシャッフル
            np.random.seed(0)               # ランダムシードを固定
            np.random.shuffle(X_train)      # シャッフル

            np.random.seed(0)
            np.random.shuffle(Y_train)

            np.random.seed(0)
            np.random.shuffle(Z_train)

            multimodal_model, X_single_model, Y_single_model, model_MM, model_X, model_Y = model_fit(X_train, Y_train, Z_train, epochs)
            calc_score(multimodal_model, X_single_model, Y_single_model, X_test, Y_test, Z_test)

            start = end + 1
            end += ref_dara_range
            epochs += 100

        MM_conf_mat = calc_score(
            multimodal_model, X_single_model, Y_single_model, X_test, Y_test, Z_test)

        MM_conf_mat = np.reshape(MM_conf_mat, (1, 5, 5))
        conf_mats = np.append(conf_mats, MM_conf_mat, axis=0)

    avg_conf_mat = np.average(conf_mats, axis=0)
    print(avg_conf_mat)

