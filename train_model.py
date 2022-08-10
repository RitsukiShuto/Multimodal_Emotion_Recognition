# Created by RitsukiShuto on 2022/08/01.
# train_model.py
#
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from secrets import choice

import keras
from keras import backend as K
from keras import Model, Input
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import save_model, load_model
from keras.layers import Activation, Dropout, AlphaDropout, Conv1D, Conv2D, Reshape, Lambda
from keras.layers import GlobalMaxPooling1D, MaxPool2D, MaxPool1D, GlobalMaxPooling2D
from keras.layers import BatchNormalization, Embedding, Concatenate, Maximum,Add


# モデル訓練用関数
def train_SetData_only(sound_labeled_X1, tfidf_labeled_X2, label):      # セットになったデータのみ学習
    print("train_SetData_only")

    X1 = sound_labeled_X1.to_numpy()        # 学習データをnumpy配列に変換
    X2 = tfidf_labeled_X2.to_numpy()
    y = np.array(label)
    y = y.astype(int)

    print("modality1:", X1.shape)           # DEBUG
    print("modality2:", X2.shape)
    print("label:", len(y))

    # データを分割
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, shuffle=True, test_size=0.15)

    print("X1_train.shape>", X1_train.shape)
    print("X2_train.shape>", X2_train.shape)

    # モデルを定義
    # 各種パラメータ
    length = len(X1_train)
    X1_dim = X1_train.shape[1]      # モダリティ1(音声)の次元数
    X2_dim = X2_train.shape[1]      # モダリティ2(テキスト)の次元数

    print("X1 dim:", X1_dim)        # DEBUG
    print("X2 dim:", X2_dim)
    print("length:", length)

    # 各モダリティの特徴量抽出層を定義
    input_X1 = Input(batch_shape=(length, X1_dim), name='input_X1')
    h11 = Dense(units=64, activation='softplus')(input_X1)
    h12 = Dense(units=32, activation='softplus')(h11)
    z1 = Dense(units=16, activation='softplus')(h12)

    input_X2 = Input(batch_shape=(length, X2_dim), name='input_X2')
    h21 = Dense(units=276, activation='softplus')(input_X2)
    h22 = Dense(units=138, activation='softplus')(h21)
    h23 = Dense(units=69, activation='softplus')(h22)
    h24 = Dense(units=34, activation='softplus')(h23)
    z2 = Dense(units=16, activation='softplus')(h24)

    # 分類層
    encoder = Concatenate()([z1, z2])
    output = Dense(units=16, activation='relu', name='last_dense')(encoder)

    classification = Dropout(0.5)(output)
    classification = Dense(units=1, activation='softmax', name='classfication_layer')(classification)

    model = Model([input_X1, input_X2], classification)                     
    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()

    # モデルの学習
    epochs = 2000       # 学習用パラメータ
    model.fit(x=[X1_train, X2_train], y=y_train, shuffle=True, batch_size=length, epochs=epochs)

    print("学習完了")

    # モデルを評価
    y_preds = model.predict(X1_test, X2_test)
    y_pred_ = np.argmax(y_preds, axis=1)
    y_test_ = np.argmax(y_test, axis=1)

    print(accuracy_score(y_test_, y_pred_))
    print(classification_report(y_test_, y_pred_))
    print(confusion_matrix(y_test_, y_pred_))


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
    label_list = []
    label_dic = {'NEU': '0', 'OTH': '1', 'ACC': '2', 'ANG': '3', 'ANT': '4', 'DIS': '5', 'FEA': '6', 'JOY': '7', 'SAD': '8', 'SUR': '9'}

    for row in meta_data.values:
        emo = str(row[9])

        if emo in label_dic:
            label = label_dic[emo]
            label_list.append(label)

    # 教師ありデータの読み込み
    sound_labeled_X1 = pd.read_csv("train_data/2div/POW_labeled.csv", header=0, index_col=0)
    tfidf_labeled_X2 = pd.read_csv("train_data/2div/TF-IDF_labeled_PCA.csv", header=0, index_col=0)


    # モードを選択
    print("\n--\nSelect a function to execute")
    print("[0]train_SetData_only\n[1]train_all_data\n[2]eval_SetData_only\n[3]eval_all_data\n")
    mode = input("imput the number:")

    # 実行する関数によって分岐
    if mode == '0':         # 教師ありデータのみで学習
        # 教師ありデータを表示
        print("\n\nsupervised sound data\n", sound_labeled_X1.head(), "\n")
        print("supervised tfidf data\n", tfidf_labeled_X2.head(), "\n")

        train_SetData_only(sound_labeled_X1, tfidf_labeled_X2, label_list)

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

        train_all_data(sound_labeled_X1, tfidf_labeled_X2, label_list, sound_un_labeled_X1, tfidf_un_labeled_X2)

    elif mode == '2':       # 教師ありデータのみのモデルを評価
        eval_SetData_only()

    elif mode == '3':       # すべてのデータのモデルを評価
        eval_all_data()

    else:
        print("error")


if __name__ == "__main__":
    main()
