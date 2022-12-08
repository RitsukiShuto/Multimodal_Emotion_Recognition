# Created by RitsukiShuto on 2022/11/23.
# supervised_learning.py
# 教師あり学習を行う
#
import numpy as np
import datetime
import os

from sklearn.model_selection import train_test_split

from modules.model import model_fit
from modules.utils import calc_score, calc_conf_mat, save_fig

def supervised_learning(X_train, Y_train, Z_train, X_test, Y_test, Z_test):
    # ログを保存
    # 現在時刻を文字列として格納
    now = datetime.datetime.now()       # 現在時刻を取得
    time = now.strftime('%Y%m%d_%H%M')
    save_dir = "train_log/supervised/" + time
    os.mkdir(save_dir)

    # データを分割
    X_train, U_train, Y_train, V_train, Z_train, W_train = train_test_split(X_train, Y_train, Z_train, shuffle=True, test_size=0.5, random_state=0, stratify=Z_train)

    print(f"\n学習データ件数:{X_train.shape[0]}\nテストデータ件数:{Y_test.shape[0]}\n")  # type: ignore

    epochs = 250
    experiment_times = 1    # 実験回数

    conf_mats = np.zeros((experiment_times, 5, 5))

    for i in range(experiment_times):
        print(f"実験回数:{i+1}/{experiment_times}")

        # 学習
        multimodal_model, X_single_model, Y_single_model, history_MM, history_X, history_Y = model_fit(X_train, Y_train, Z_train, epochs)

        # lossとaccuracyのグラフを保存
        save_fig(save_dir, multimodal_model, history_MM, i+1, cnt=0)

        # 精度を計算
        MM_conf_mat = calc_score(multimodal_model, X_single_model, Y_single_model, X_test, Y_test, Z_test)

        # 混同行列を計算
        MM_conf_mat = np.reshape(MM_conf_mat, (1, 5, 5))
        conf_mats[i, :, :] = MM_conf_mat

    calc_conf_mat(conf_mats, experiment_times, save_dir)
