# Created by RitsukiShuto on 2022/11/23.
# main.py
# 教師あり学習、半教師あり学習を呼び出す
#
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import supervised_learning, semi_supervised_learning

def main():
    # データを読み込む
    meta_data = pd.read_csv("train_data/meta_data/MOY_mixed_meta.csv", header=0)
    meta_data = meta_data.dropna(subset=['emotion'], axis=0)      # 全体のメタデータから教師ありデータのみを抽出

    # ラベル整形
    ohe = OneHotEncoder(sparse=False)
    labels = ohe.get_feature_names(['emotion'])
    ohe = ohe.fit_transform(meta_data[['emotion']].values)
    labels = pd.DataFrame(ohe, columns=labels, dtype=np.int8)

    # 教師ありデータを読み込む
    labeled_X = pd.read_csv("train_data/mixed/MFCC_labeled.csv", header=0, index_col=0)
    labeled_Y = pd.read_csv("train_data/mixed/TF-IDF_labeled.csv", header=0, index_col=0)

    # ndarrayに変換
    X = labeled_X.to_numpy()
    Y = labeled_Y.to_numpy()
    Z = labels.to_numpy()

    # データを分割
    X_train, X_text, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z, test_size=0.2, random_state=0, stratify=0)

    # 実行するモードを選択
    print("\n--\n実行する学習を選択")
    print("[0]supervised_learning\n[1]semi_supervised_learning\n[2]evaluate_model\n")
    choice = input("imput the number:")

    if choice == '0':       # 教師あり学習
        print("run supervised learning")
        supervised_learning()

    elif choice == '1':     # 半教師あり学習
        print("run semi supervised learning")
        semi_supervised_learning()

    else:       # Error
        print("error")
        main()

if __name__ == "__main":
    main()