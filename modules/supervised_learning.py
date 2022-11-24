# Created by RitsukiShuto on 2022/11/23.
# supervised_learning.py
# 教師あり学習を行う
#
from modules.model import model_fit
from modules.evaluate_model import calc_score

def supervised_learning(X_train, Y_train, Z_train, X_test, Y_test, Z_test):
    print(f"学習データ件数:{X_train.shape[0]}\nテストデータ件数:{Y_test.shape[0]}")

    epochs = 250

    for i in range(1):
        print(f"{i}/10")

        multimodal_model, X_single_model, Y_single_model, model_MM, model_X, model_Y = model_fit(X_train, Y_train, Z_train, epochs)
        conf_mat = calc_score(multimodal_model, X_single_model, Y_single_model, X_test, Y_test, Z_test)

