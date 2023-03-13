# Created by RitsukiShuto on 2022/11/23.
# main.py
# データの読み込みを行い、教師あり学習・半教師あり学習を呼び出す
#
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from modules.supervised_learning import supervised_learning
from modules.semi_supervised_learning import semi_supervised_learning

# ラベルごとのデータ数を確認
def conut_labels(label_data):
    label_cnt = pd.DataFrame(label_data, columns=['ANG', 'JOY', 'NEU', 'SAD', 'SUR'])

    print("ANG", len(label_cnt.query('ANG == 1')))
    print("JOY", len(label_cnt.query('JOY == 1')))
    print("NEU", len(label_cnt.query('NEU == 1')))
    print("SAD", len(label_cnt.query('SAD == 1')))
    print("SUR", len(label_cnt.query('SUR == 1')))

def main():
    # データを読み込む
    meta_data = pd.read_csv("train_data/meta_data/1_meta.csv", header=0)
    meta_data = meta_data.dropna(subset=['emotion'], axis=0)      # 全体のメタデータから教師ありデータのみを抽出

    # ラベル付きデータ
    labeled_X = pd.read_csv("train_data/feature_vector/1_POW_labeled.csv", header=0, index_col=0)
    labeled_Y = pd.read_csv("train_data/feature_vector/1_TF-IDF_labeled.csv", header=0, index_col=0)

    # ラベルなしデータ
    un_labeled_U = pd.read_csv("train_data/feature_vector/1_POW_un_labeled.csv", header=0, index_col=0)
    un_labeled_V = pd.read_csv("train_data/feature_vector/1_TF-IDF_un_labeled.csv", header=0, index_col=0)

    # ラベル整形
    ohe = OneHotEncoder(sparse=False)
    encoded = ohe.fit_transform(meta_data[['emotion']].values)
    label = ohe.get_feature_names(['emotion'])
    label_list = pd.DataFrame(encoded, columns=label, dtype=np.int8)

    # ndarrayに変換
    X = labeled_X.to_numpy()
    Y = labeled_Y.to_numpy()
    U = un_labeled_U.to_numpy()
    V = un_labeled_V.to_numpy()
    Z = label_list.to_numpy()

    # データを分割
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z, shuffle=True, test_size=488, random_state=0, stratify=Z)

    epochs = 500
    batch_size = 256
    experiment_times = 3       # 実験回数

    # 実行するモードを選択
    print("\n--\n実行する学習を選択")
    print("[1]supervised_learning\n[2]semi_supervised_learning\n[3]DBG supervised_learning\n[4]DBG semi_supervised_learning\n")
    choice = input("imput the number:")

    if choice == '1':       # 教師あり学習
        print("run supervised learning")

        # データを分割
        # FIXME: 半教師あり学習の初期学習データ数と揃えるときはコメントアウトを解除せよ
        #X_train, U_train, Y_train, V_train, Z_train, W_train = train_test_split(X_train, Y_train, Z_train, shuffle=True, test_size=0.7, random_state=0, stratify=Z_train)

        print(f"ラベル付きデータ:{X_train.shape[0]}")
        print(f"テストデータ:{len(Z_test)}")

        print("\n学習データのクラスごとの件数")
        conut_labels(Z_train)

        print("\nテストデータのクラスごとの件数")
        conut_labels(Z_test)

        supervised_learning(X_train, Y_train, Z_train, X_test, Y_test, Z_test,
                            epochs, batch_size, experiment_times
                            )

    elif choice == '2':     # 半教師あり学習
        print("run semi supervised learning")

        X_train, U_un_labeled, Y_train, V_un_labeled, Z_train, W_train = train_test_split(X_train, Y_train, Z_train, shuffle=True, test_size=0.7, random_state=0, stratify=Z_train)

        # ラベルなしデータと結合
        U_train = np.append(U_un_labeled, U, axis=0)
        V_train = np.append(V_un_labeled, V, axis=0)

        print(f"ラベル付きデータ:{X_train.shape[0]}")
        print(f"ラベルなしデータ:{U_train.shape[0]}")
        print(f"テストデータ:{len(Z_test)}")

        print("\n学習データのクラスごとの件数")
        conut_labels(Z_train)

        print("\nテストデータのクラスごとの件数")
        conut_labels(Z_test)

        semi_supervised_learning(X_train, Y_train, U_train, V_train,
                                 W_train, Z_train, X_test, Y_test, Z_test,
                                 epochs, batch_size, experiment_times
                                 )

    # ここからの処理はデバッグ用である。プログラム全体の挙動をチェックしたいときはこれを実行せよ
    elif choice == '3':
        print("[DBG] supervised learning")

        epochs = 1

        supervised_learning(X_train, Y_train, Z_train, X_test, Y_test, Z_test,
                            epochs, batch_size, experiment_times
                            )

    elif choice == '4':
        print("[DBG] semi supervised learning")

        X_train, U_un_labeled, Y_train, V_un_labeled, Z_train, W_train = train_test_split(X_train, Y_train, Z_train, shuffle=True, test_size=0.7, random_state=0, stratify=Z_train)

        # ラベルなしデータと結合
        U_train = np.append(U_un_labeled, U, axis=0)
        V_train = np.append(V_un_labeled, V, axis=0)

        epochs = 1

        semi_supervised_learning(X_train, Y_train, U_train, V_train,
                                 W_train, Z_train, X_test, Y_test, Z_test,
                                 epochs, batch_size, experiment_times
                                )

    else:       # Error
        print("error")
        main()

if __name__ == "__main__":
    main()