# Created by RitsukiShuto on 2022/11/23.
# supervised_learning.py
# 教師あり学習を行う
# このプログラムはmain.pyから呼び出される。これ単体で実行しても動かない。
#
import numpy as np
import datetime
import os

from sklearn.model_selection import train_test_split

from modules.model import model_compile, model_fit
from modules.utils import calc_score, calc_conf_mat, save_fig


def supervised_learning(X_train, Y_train, Z_train, X_test, Y_test, Z_test,
                        epochs, batch_size, experiment_times):
    # ログを保存
    # 現在時刻を文字列として格納
    now = datetime.datetime.now()       # 現在時刻を取得
    
    time = now.strftime('%Y%m%d_%H%M')
    save_dir = "train_log/supervised/" + time
    os.mkdir(save_dir)

    print(f"\n学習データ件数:{X_train.shape[0]}\nテストデータ件数:{Y_test.shape[0]}\n")  # type: ignore

    conf_mats = np.zeros((experiment_times, 5, 5))

    for i in range(experiment_times):
        print(f"実験回数:{i+1}/{experiment_times}")

        # モデル生成
        multimodal_model, X_single_model, Y_single_model = model_compile(X_train, Y_train)

        # 学習
        history_MM, history_X, history_Y = model_fit(multimodal_model, X_single_model, Y_single_model,
                                                     X_train, Y_train, Z_train, epochs, batch_size)

        # 精度を計算
        MM_conf_mat, accuracy = calc_score(multimodal_model, X_single_model, Y_single_model, X_test, Y_test, Z_test)
        df1, df2 = calc_conf_mat(MM_conf_mat, None)

        # lossとaccuracyのグラフを保存
        save_fig(save_dir, multimodal_model, history_MM, None, df1, df2, i+1, 0)

        # 混同行列を計算
        MM_conf_mat = np.reshape(MM_conf_mat, (1, 5, 5))
        conf_mats[i, :, :] = MM_conf_mat

    df1, df2 = calc_conf_mat(conf_mats, experiment_times)       # 混同行列の平均を求める
    save_fig(save_dir, multimodal_model, history_MM, None, df1, df2, 'score', 0)    # ログを保存
