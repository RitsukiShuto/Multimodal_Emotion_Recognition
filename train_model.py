# Created by RitsukiShuto on 2022/08/01.
# train_model.py
#
from gc import callbacks
from tabnanny import verbose
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from secrets import choice

import keras
from keras import Model, Input
from tensorflow.keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.models import save_model, load_model
from keras.layers import Dense, Dropout, Concatenate

from keras.utils.vis_utils import plot_model

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

import datetime
import os

# セクションごとにニューラルネットワークを定義
# 教師あり・半教師あり学習両用
def X1_feature_quantity_extracting_layer(length, X1_dim):
    # モダリティ1の特徴量抽出層
    input_X1 = Input(batch_shape=(length, X1_dim), name='input_X1')
    h11 = Dense(units=64, activation='relu')(input_X1)
    h12 = Dense(units=32, activation='relu')(h11)
    z1 = Dense(units=4, activation='relu')(h12)

    # 単一モダリティでの分類用のネットワーク
    c_x1 = Dropout(0.5)(z1)
    z12 = Dense(units=8, activation='softmax')(c_x1)
    x1_single_model = Model(input_X1, z12)

    return input_X1, z1, x1_single_model

def X2_feature_quantity_extracting_layer(length, X2_dim):
    # モダリティ2の特徴量抽出層
    input_X2 = Input(batch_shape=(length, X2_dim), name='input_X2')
    h21 = Dense(units=276, activation='relu')(input_X2)
    h22 = Dense(units=69, activation='relu')(h21)
    h23 = Dense(units=34, activation='relu')(h22)
    z2 = Dense(units=4, activation='relu')(h23)

    # 単一モダリティでの分類用のネットワーク
    c_x2 = Dropout(0.5)(z2)
    z22 = Dense(units=8, activation='softmax')(c_x2)
    x2_single_model = Model(input_X2, z22)

    return input_X2, z2, x2_single_model

def classification_layer(input_X1, input_X2, z1, z2):
    # 特徴量を合成
    concat =  Concatenate()([z1, z2])

    # 分類層
    c1 = Dense(units=8, activation='relu', name='classification_1')(concat)
    c2 = Dense(units=16, activation='relu', name='classification_2')(c1)
    #c3 = Dense(units=8, activation='relu', name='classification_3')(c2)

    # 出力層
    classification = Dropout(0.5)(c2)
    output = Dense(units=8, activation='softmax', name='classfication_layer')(classification)

    multimodal_model = Model([input_X1, input_X2], output)

    return multimodal_model

# 教師あり学習
def train_SetData_only(sound_labeled_X1, tfidf_labeled_X2, label):      # セットになったデータのみ学習
    print("train_SetData_only")

    # TODO: dataframeを最初からNumpyArrayで読み込むように変更
    X1 = sound_labeled_X1.to_numpy()        # 学習データをnumpy配列に変換
    X2 = tfidf_labeled_X2.to_numpy()
    y = label.to_numpy()

    # データを分割
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, shuffle=True, test_size=0.15, random_state=0)
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1_train, X2_train, y_train, shuffle=True, test_size=0.15, random_state=0)

    # モデルを定義
    # 各種パラメータ
    length = len(X1_train)          # 学習データの数
    X1_dim = X1_train.shape[1]      # モダリティ1(音声)の次元数
    X2_dim = X2_train.shape[1]      # モダリティ2(テキスト)の次元数

    # TODO: 出力するものを吟味したほうがいい
    print("X1_train.shape:", X1_train.shape)
    print("X2_train.shape:", X2_train.shape)
    print("y_train.shape:", y_train.shape)
    print("X1 dim:", X1_dim)
    print("X2 dim:", X2_dim)
    print("length:", length)

    # 特徴量抽出層
    input_X1, z1, x1_single_model = X1_feature_quantity_extracting_layer(length, X1_dim)
    input_X2, z2, x2_single_model = X2_feature_quantity_extracting_layer(length, X2_dim)

    # 分類層
    multimodal_model = classification_layer(input_X1, input_X2, z1, z2)

    # モデル生成
    multimodal_model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                             loss=categorical_crossentropy,
                             metrics=['accuracy'])

    x1_single_model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                            loss=categorical_crossentropy,
                            metrics=['accuracy'])

    x2_single_model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                            loss=categorical_crossentropy,
                            metrics=['accuracy'])
    
    # ----------------------------------------------------------------------------------------
    # モデルの学習
    epochs = 200       # 学習用パラメータ
    batch_size = 8

    multimodal_fit = multimodal_model.fit(x=[X1_train, X2_train], y=y_train,
                                          validation_data=([X1_val, X2_val], y_val),
                                          batch_size=batch_size, epochs=epochs)

    x1_fit = x1_single_model.fit(x=X1_train, y=y_train,
                                 validation_data=(X1_val, y_val),
                                 batch_size = batch_size, epochs=epochs)

    x2_fit = x2_single_model.fit(x=X2_train, y=y_train,
                                 validation_data=(X2_val, y_val),
                                 batch_size = batch_size, epochs=epochs)
                                 
    # TODO: 学習済みモデルを保存するように変更


    # -----------------------------------------------------------------------------------------
    # モデルを評価
    result_multimodal = multimodal_model.predict(x=[X1_test, X2_test])
    result_multimodal_pred = np.argmax(result_multimodal, axis=1)
    result_multimodal_test = np.argmax(y_test, axis=1)

    reslut_x1 = x1_single_model.predict(x=X1_test)
    result_x1_pred = np.argmax(reslut_x1, axis=1)
    result_x1_test = np.argmax(y_test, axis=1)

    reslut_x2 = x2_single_model.predict(x=X2_test)
    result_x2_pred = np.argmax(reslut_x2, axis=1)
    result_x2_test = np.argmax(y_test, axis=1)

    # 結果を表示
    print("classification report multimodal\n", classification_report(result_multimodal_test, result_multimodal_pred))
    print("classification report x1\n", classification_report(result_x1_test, result_x1_pred))
    print("classification report x2\n", classification_report(result_x2_test, result_x2_pred))

    # 結果を保存
    supervised_train_save_log(multimodal_model, x1_single_model, x2_single_model, multimodal_fit, x1_fit, x2_fit)


def train_all_data(sound_labeled_X1, tfidf_labeled_X2, label, sound_un_labeled_X1, tfidf_un_labeled_X2):          # すべてのデータで学習
    print("train_all_data")

# 教師あり学習のログを保存
def supervised_train_save_log(multimodal_model, x1_single_model, x2_single_model, multimodal_fit, x1_fit, x2_fit):
    now = datetime.datetime.now()                                   # 現在時刻を取得(YYYYMMDD_hhmm)
    make_dir = "./train_log/" +  now.strftime('%Y%m%d_%H%M')
    os.mkdir(make_dir)                                              # 現在時刻のディレクトリを作成

    # モデルの構成を保存(.png)
    plot_model(multimodal_model, to_file=make_dir + '/MM_model_shape' + now.strftime('%Y%m%d_%H%M') + '.png')
    plot_model(x1_single_model, to_file=make_dir + '/x1_model_shape' + now.strftime('%Y%m%d_%H%M') + '.png')
    plot_model(x2_single_model, to_file=make_dir + '/x2_model_shape' + now.strftime('%Y%m%d_%H%M') + '.png')

    # 学習ログ(.csv)
    df1 = pd.DataFrame(multimodal_fit.history)      # DataFrame化
    df2 = pd.DataFrame(x1_fit.history)
    df3 = pd.DataFrame(x2_fit.history)

    df1.to_csv(make_dir + '/MM_train_log' + now.strftime('%Y%m%d_%H%M') + '.csv')     # csvで保存
    df2.to_csv(make_dir + '/x1_train_log' + now.strftime('%Y%m%d_%H%M') + '.csv')
    df3.to_csv(make_dir + '/x2_train_log' + now.strftime('%Y%m%d_%H%M') + '.csv')

    # グラフ(.png)
    fig = plt.figure()

    loss_ylabel = 'Loss'
    acc_ylabel = 'Accuracy'

    ax1 = fig.add_subplot(3, 2, 1)      # multimodal loss
    ax2 = fig.add_subplot(3, 2, 2)      # multimodal acc
    ax3 = fig.add_subplot(3, 2, 3)      # x1 loss
    ax4 = fig.add_subplot(3, 2, 4)      # x1 acc
    ax5 = fig.add_subplot(3, 2, 5)      # x2 loss 
    ax6 = fig.add_subplot(3, 2, 6)      # x2 acc

    ax1.plot(multimodal_fit.history['loss'])
    ax1.plot(multimodal_fit.history['val_loss'])
    ax1.set_title('multimodal loss')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    ax2.plot(multimodal_fit.history['accuracy'])
    ax2.plot(multimodal_fit.history['val_accuracy'])
    ax2.set_title('multimodal accuracy')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    ax3.plot(x1_fit.history['loss'])
    ax3.plot(x1_fit.history['val_loss'])
    ax3.set_title('x1 loss')
    ax3.legend(['Train', 'Validation'], loc='upper left')

    ax4.plot(x1_fit.history['accuracy'])
    ax4.plot(x1_fit.history['val_accuracy'])
    ax4.set_title('x1 accuracy')
    ax4.legend(['Train', 'Validation'], loc='upper left')

    ax5.plot(x2_fit.history['loss'])
    ax5.plot(x2_fit.history['val_loss'])
    ax5.set_title('x2 loss')
    ax5.legend(['Train', 'Validation'], loc='upper left')

    ax6.plot(x2_fit.history['accuracy'])
    ax6.plot(x2_fit.history['val_accuracy'])
    ax6.set_title('x2 accuracy')
    ax6.legend(['Train', 'Validation'], loc='upper left')

    fig.tight_layout()
    plt.savefig(make_dir + "/reslt_graph" + now.strftime('%Y%m%d_%H%M') + '.png')
    plt.show()

def main():
    # メタデータのディレクトリ
    meta_data = pd.read_csv("data/supervised_list.csv", header=0)
    supervised_meta = meta_data.dropna(subset=['emotion'], axis=0)      # 全体のメタデータから教師ありデータのみを抽出

    # ラベルを読み込み
    # ワンホットエンコーディング
    ohe = OneHotEncoder(sparse=False)
    encoded = ohe.fit_transform(supervised_meta[['emotion']].values)

    label = ohe.get_feature_names(['emotion'])
    label_list = pd.DataFrame(encoded, columns=label, dtype=np.int8)
    
    # 教師ありデータの読み込み
    sound_labeled_X1 = pd.read_csv("train_data/2div/POW_labeled.csv", header=0, index_col=0)
    tfidf_labeled_X2 = pd.read_csv("train_data/2div/TF-IDF_labeled_PCA.csv", header=0, index_col=0)


    # モードを選択
    print("\n--\nSelect a function to execute")
    print("[0]train_SetData_only\n[1]train_all_data\n")
    mode = input("imput the number:")

    # 実行する関数によって分岐
    if mode == '0':         # 教師ありデータのみで学習
        # 教師ありデータとラベルを表示
        print("\n\nsupervised sound data\n", sound_labeled_X1.head(), "\n")
        print("supervised tfidf data\n", tfidf_labeled_X2.head(), "\n")
        print("label data\n", label_list.head(), "\n")

        # 教師あり学習を実行
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

    else:
        print("error")


if __name__ == "__main__":
    main()
