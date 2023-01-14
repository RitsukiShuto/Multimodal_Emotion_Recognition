# Created by RitsukiShuto on 2022/11/23.
# main.py
# 教師あり学習、半教師あり学習を呼び出す
#
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from modules.supervised_learning import supervised_learning
from modules.semi_supervised_learning import semi_supervised_learning

def main():
    # データを読み込む
    meta_data = pd.read_csv("train_data/meta_data/3_meta.csv", header=0)
    meta_data = meta_data.dropna(subset=['emotion'], axis=0)      # 全体のメタデータから教師ありデータのみを抽出

    # ラベル整形
    ohe = OneHotEncoder(sparse=False)
    encoded = ohe.fit_transform(meta_data[['emotion']].values)
    label = ohe.get_feature_names(['emotion'])
    label_list = pd.DataFrame(encoded, columns=label, dtype=np.int8)

    # 教師ありデータを読み込む
    labeled_X = pd.read_csv("train_data/mixed/3_POW_labeled.csv", header=0, index_col=0)
    labeled_Y = pd.read_csv("train_data/mixed/3_TF-IDF_labeled.csv", header=0, index_col=0)

    # ndarrayに変換
    X = labeled_X.to_numpy()
    Y = labeled_Y.to_numpy()
    Z = label_list.to_numpy()
    
    label_cnt = pd.DataFrame(Z, columns=['ANG', 'JOY', 'NEU', 'SAD', 'SUR'])    

    print("\n学習データのクラスごとの件数")
    print("ANG", len(label_cnt.query('ANG == 1')))
    print("JOY", len(label_cnt.query('JOY == 1')))
    print("NEU", len(label_cnt.query('NEU == 1')))
    print("SAD", len(label_cnt.query('SAD == 1')))
    print("SUR", len(label_cnt.query('SUR == 1')))

    # データを分割
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z, shuffle=True, test_size=488, random_state=0, stratify=Z)

    epochs = 500
    batch_size = 256
    experiment_times = 10

    # 実行するモードを選択
    print("\n--\n実行する学習を選択")
    print("[1]supervised_learning\n[2]semi_supervised_learning\n[3]DBG supervised_learning\n[4]DBG semi_supervised_learning\n")
    choice = input("imput the number:")

    if choice == '1':       # 教師あり学習
        print("run supervised learning")
        supervised_learning(X_train, Y_train, Z_train, X_test, Y_test, Z_test,
                            epochs, batch_size, experiment_times
                            )

    elif choice == '2':     # 半教師あり学習
        print("run semi supervised learning")
        semi_supervised_learning(X_train, Y_train, Z_train, X_test, Y_test, Z_test,
                                 epochs, batch_size, experiment_times
                                 )

    elif choice == '3':
        epochs = 1
        print("[DBG] supervised learning")
        supervised_learning(X_train, Y_train, Z_train, X_test, Y_test, Z_test,
                            epochs, batch_size, experiment_times
                            )

    elif choice == '4':
        epochs = 1
        print("[DBG] semi supervised learning")
        semi_supervised_learning(X_train, Y_train, Z_train, X_test, Y_test, Z_test,
                                 epochs, batch_size, experiment_times
                                )

    else:       # Error
        print("error")
        main()

if __name__ == "__main__":
    main()