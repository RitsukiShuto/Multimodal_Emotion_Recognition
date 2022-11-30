# Created by RitsukiShuto on 2022/11/23.
# eval_model.py
# テストデータでモデルのテストを行う
#
import numpy as np

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

    return MM_conf_mat

def calc_conf_mat(conf_mats):
    avg_conf_mat = np.average(conf_mats, axis=0)
