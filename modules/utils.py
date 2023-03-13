# Created by RitsukiShuto on 2022/11/23.
# utils.py
# テストデータでモデルのテストを行う
#
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix

def calc_score(model_MM, model_X, model_Y, X_test, Y_test, Z_test):
    # テストデータで推定する
    pred_MM = model_MM.predict(x=[X_test, Y_test])
    pred_X1 = model_X.predict(X_test)
    pred_X2 = model_Y.predict(Y_test)

    MM_pred_ = np.argmax(pred_MM, axis=1)
    X1_pred_ = np.argmax(pred_X1, axis=1)
    X2_pred_ = np.argmax(pred_X2, axis=1)

    Z_test_ = np.argmax(Z_test, axis=1)

    # クラスごとの分類精度を表示する。
    MM_conf_mat = confusion_matrix(Z_test_, MM_pred_)
    X_conf_mat = confusion_matrix(Z_test_, X1_pred_)
    Y_conf_mat = confusion_matrix(Z_test_, X2_pred_)

    # 精度を表示
    MM_score = model_MM.evaluate(x=[X_test, Y_test], y=Z_test, verbose=0)
    X_score = model_X.evaluate(X_test, Z_test, verbose=0)
    Y_score = model_Y.evaluate(Y_test, Z_test, verbose=0)

    print(f"\nマルチモーダルモデル\ntest loss:{MM_score[0]}\ntest accuracy:{MM_score[1]}\n\n")
    print(f"X 単一モデル\ntest loss:{X_score[0]}\ntest accuracy:{X_score[1]}\n\n")
    print(f"Y 単一モデル\ntest loss:{Y_score[0]}\ntest accuracy:{Y_score[1]}\n\n")

    print("マルチモーダル混同行列")
    print(MM_conf_mat)

    return MM_conf_mat, MM_score[1]

def calc_conf_mat(conf_mats, experiment_times):
    labels = ['ANG', 'JOY', 'NEU', 'SAD', 'SUR']

    # 確率を計算
    if experiment_times != None:
        prob_conf_mats = np.zeros((experiment_times, 5, 5))

        for i in range(experiment_times):
            for j in range(5):
                for k in range(5):
                    prob_conf_mats[i][j][k] = conf_mats[i][j][k] / sum(conf_mats[i, j, :])

        prob_avg_conf_mat = np.average(prob_conf_mats, axis=0)
        num_avg_conf_mat = np.average(conf_mats, axis=0)
    
    else:
        prob_conf_mats = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                prob_conf_mats[i][j] = conf_mats[i][j] / sum(conf_mats[i, :])

        prob_avg_conf_mat = prob_conf_mats
        num_avg_conf_mat = conf_mats

    # 精度と平均を追加
    accuracy = np.trace(num_avg_conf_mat) / np.sum(num_avg_conf_mat)
    average = np.trace(prob_avg_conf_mat) / 5

    # 実験結果を保存
    df1 = pd.DataFrame(prob_avg_conf_mat, columns=labels, index=labels)
    df2 = pd.DataFrame(num_avg_conf_mat, columns=labels, index=labels)

    df1.loc['average', 'score'] = average
    df1.loc['accuracy', 'score'] = accuracy

    return df1, df2

def save_fig(save_dir, multimodal_model, history_MM, accuracy_trend, df1, df2, num, cnt):
    if num == 1:        # ループ1回目のときだけモデルの構成を保存
        plot_model(multimodal_model, to_file=save_dir + '/model_shape_MM.png', show_shapes=True)

    if cnt != 0:        # cnt == 0のときは教師あり学習
        f_num = str(num) + '-' + str(cnt)
    else:
        f_num = num

    if not os.path.exists(save_dir + "/" + str(num)):
        os.mkdir(save_dir + "/" + str(num))

    # 学習ログ(.csv)
    df = pd.DataFrame(history_MM.history)      # DataFrame化
    df.to_csv(save_dir + "/" + str(num) + '/train_log_MM' + str(f_num) + '.csv')     # csvで保存

    # グラフ(.png)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(2, 1, 1)      # multimodal loss
    ax2 = fig1.add_subplot(2, 1, 2)      # multimodal acc

    ax1.plot(history_MM.history['loss'])
    ax1.set_title('multimodal loss')
    ax1.legend(['Train'], loc='upper left')

    ax2.plot(history_MM.history['accuracy'])
    ax2.set_title('multimodal accuracy')
    ax2.legend(['Train'], loc='upper left')

    plt.savefig(save_dir + "/" + str(num) + "/MM_fig" + str(f_num) + ".png")

    # 混同行列を保存
    df1.to_csv(save_dir + "/" + str(num) + '/prob_conf_mat' + str(f_num) + '.csv')
    df2.to_csv(save_dir + "/" + str(num) + '/num_conf_mat' + str(f_num) + '.csv')

    if accuracy_trend != None:
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(accuracy_trend)

        plt.savefig(save_dir + "/" + str(num) + "/accuracy_trend" + str(f_num) + ".png")
