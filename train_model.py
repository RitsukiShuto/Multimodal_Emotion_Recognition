# Created by RitsukiShuto on 2022/08/01.
# train_model.py
#
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder
from secrets import choice


# モデル訓練用関数
def train_SetData_only(sound_labeled_X1, tfidf_labeled_X2, label):      # セットになったデータのみ学習
    print("train_SetData_only")

    X1 = list(sound_labeled_X1)
    X2 = list(tfidf_labeled_X2)
    
    y = list(label)

    # 学習データとテストデータに分割


    # モデルを評価
    eval_SetData_only()


def train_all_data(sound_labeled_X1, tfidf_labeled_X2, label, sound_un_labeled_X1, tfidf_un_labeled_X2):          # すべてのデータで学習
    print("train_all_data")


# モデル評価用関数
def eval_SetData_only():       # train_SetData_only()モデルの評価
    print("eval_SetData_only")


def eval_all_data():           # train_all_data()モデルの評価
    print("eval_all_data")


def main():
    # メタデータのディレクトリ
    meta_data = pd.read_csv("data/supervised_list.csv", header=0)

    # ラベルを読み込み
    # ラベルを数値に変換
    # ['ACC', 'ANG', 'ANT', 'DIS', 'FEA', 'JOY', 'SAD', 'SUR', 'NEU', 'OTH'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    label = 0

    # 教師ありデータの読み込み
    sound_labeled_X1 = pd.read_csv("train_data/2div/POW_labeled.csv", header=0, index_col=0)
    tfidf_labeled_X2 = pd.read_csv("train_data/2div/TF-IDF_labeled_PCA.csv", header=0, index_col=0)


    # モードを選択
    print("Select a function to execute")
    print("[0]train_SetData_only\n[1]train_all_data\n[2]eval_SetData_only\n[3]eval_all_data\n")
    mode = input("imput the number:")

    if mode == '0':         # 教師ありデータのみで学習
        # 教師ありデータを表示
        print("\n\nsupervised sound data\n", sound_labeled_X1.head(), "\n")
        print("supervised tfidf data\n", tfidf_labeled_X2.head(), "\n")

        train_SetData_only(sound_labeled_X1, tfidf_labeled_X2, label)

    elif mode == '1':       # すべてのデータで学習
        # 教師なしデータを読み込み
        sound_un_labeled_X1 = pd.read_csv("train_data/2div/POW_un_labeled.csv", header=0, index_col=0)
        tfidf_un_labeled_X2 = pd.read_csv("train_data/2div/TF-IDF_un_labeled_PCA.csv", header=0, index_col=0)    

        # データをランダムに欠損させる<試作版>
        for (unX1_row, unX2_row) in zip(sound_un_labeled_X1.values, tfidf_un_labeled_X2.values):
            missing = random.choice([0, 1])

            if missing == 0:
                unX1_row[:] = None

            else:
                unX2_row[:] = None

        # 教師ありデータを表示
        print("\n\nsupervised sound data\n", sound_labeled_X1.head(), "\n")
        print("supervised tfidf data\n", tfidf_labeled_X2.head(), "\n")

        # 教師なしデータを表示
        print("\nun supervised sound data\n", sound_un_labeled_X1.head(), "\n")
        print("un supervised tfidf data\n", tfidf_un_labeled_X2.head(), "\n")

        # データの欠損数を表示
        print("missing sound data:", sound_un_labeled_X1.isnull().sum().sum() / 128)
        print("missing tfidf data:", tfidf_un_labeled_X2.isnull().sum().sum() / 553, "\n")

        train_all_data(sound_labeled_X1, tfidf_labeled_X2, label, sound_un_labeled_X1, tfidf_un_labeled_X2)

    elif mode == '2':       # 教師ありデータのみのモデルを評価
        eval_SetData_only()

    elif mode == '3':       # すべてのデータのモデルを評価
        eval_all_data()

    else:
        print("error")


if __name__ == "__main__":
    main()
