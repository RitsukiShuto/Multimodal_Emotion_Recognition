# Created by RitsukiShuto on 2022/11/23.
# supervised_learning.py
# 半教師あり学習を行う
# このプログラムはmain.pyから呼び出される。これ単体で実行しても動かない。
#
import pandas as pd
import numpy as np
import datetime
import os

from sklearn.model_selection import train_test_split

from modules.model import model_compile, model_fit
from modules.utils import calc_score, calc_conf_mat, save_fig

def semi_supervised_learning(X_train, Y_train, U_train, V_train, W_train,
                             Z_train, X_test, Y_test, Z_test,
                             epochs, batch_size, experiment_times):
    # ログ保存用
    # 現在時刻を文字列として格納
    now = datetime.datetime.now()       # 現在時刻を取得
    time = now.strftime('%Y%m%d_%H%M')
    save_dir = "train_log/semi_supervised/" + time
    os.mkdir(save_dir)

    # TODO: ラベルなしデータのみを扱う際は以下の3行をコメントアウトせよ
    #U_train = un_labeled_U
    #V_train = un_labeled_V
    #W_train = []

    print(f"\n教師ありデータ:{X_train.shape[0]}\n教師なしデータ:{U_train.shape[0]}\nテストデータ:{X_test.shape[0]}\n")  # type: ignore

    # ループ回数等に関わる変数
    data_cnt = U_train.shape[0]                 # データ件数
    ref_data_range = 60                         # 推定する未ラベルデータの個数
    loop_times = data_cnt / ref_data_range      # ループ回数

    conf_mats = np.zeros((experiment_times, 5, 5))

    for i in range(experiment_times):
        start = 0                       # 未ラベルデータの参照範囲[start]
        end = ref_data_range - 1        # 未ラベルデータの参照範囲[end] --> [参照するデータ数] - 1が最後のインデックス
        data_count_to_add = -20         # 仮ラベル付けするデータの件数
        accuracy_trend = []             # 精度の推移を格納するための変数
    
        print(f"実験回数:{i+1}/{experiment_times}")
        print("初回学習")

        # モデル生成
        multimodal_model, X_single_model, Y_single_model = model_compile(X_train, Y_train)

        # 1回目のときだけmodel summaryを表示
        if i == 0:
            print(multimodal_model.summary())

        # 学習
        fit_multimodal_model, fit_X_single_model, fit_Y_single_model = model_fit(multimodal_model, X_single_model, Y_single_model,
                                                                                 X_train, Y_train, Z_train, epochs, batch_size)

        # 未知データでテスト
        calc_score(multimodal_model, X_single_model, Y_single_model, X_test, Y_test, Z_test)

        # ラベルなしデータを推定して仮ラベルを付与する
        for j in range(int(loop_times)+1):
            print(f"{j+1}/{int(loop_times)+1}")
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

            np.random.seed(0)               # ランダムシードを固定
            np.random.shuffle(X_train)      # シャッフル

            np.random.seed(0)
            np.random.shuffle(Y_train)

            np.random.seed(0)
            np.random.shuffle(Z_train)

            # 未ラベルデータの参照範囲を更新
            start = end + 1

            if j+1 == int(loop_times):
                end = data_cnt      # 最後のループのときは[データ長-1]の値をendにする

                # FIXME: 切り上げで計算するように修正する
                data_count_to_add = int(-(end - start) / 3)
                if data_count_to_add == 0:
                    data_count_to_add = -1
            else:
                end += ref_data_range

            # 学習
            fit_multimodal_model, fit_X_single_model, fit_Y_single_model = model_fit(multimodal_model, X_single_model, Y_single_model,
                                                                                    X_train, Y_train, Z_train, epochs, batch_size)

            # テスト
            conf_mat, accuracy = calc_score(multimodal_model, X_single_model, Y_single_model, X_test, Y_test, Z_test)

            accuracy_trend.append(accuracy)

            # lossとaccuracyのグラフを保存
            df1, df2 = calc_conf_mat(conf_mat, None)
            save_fig(save_dir, multimodal_model, fit_multimodal_model, None, df1, df2, i+1, j+1)

        MM_conf_mat, accuracy = calc_score(multimodal_model, X_single_model, Y_single_model, X_test, Y_test, Z_test)
        save_fig(save_dir, multimodal_model, fit_multimodal_model, accuracy_trend, df1, df2, i+1, 0)

        MM_conf_mat = np.reshape(MM_conf_mat, (1, 5, 5))
        conf_mats[i, :, :] = MM_conf_mat

    df1, df2 = calc_conf_mat(conf_mats, experiment_times)
    save_fig(save_dir, multimodal_model, fit_multimodal_model, None, df1, df2, 'score', 0)
