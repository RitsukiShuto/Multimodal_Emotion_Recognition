# Created by RitsukiShuto on 2022/08/01.
# train_model.py
#
from secrets import choice

import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras import Model, Input
from tensorflow.keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.models import save_model, load_model
from keras.layers import Dense, Dropout, Concatenate, MaxPooling1D
from keras.utils.vis_utils import plot_model

# エンコーダ
def X1_encoder(X1_dim):
    # モダリティ1の特徴量抽出層
    input_X1 = Input(shape=(None, X1_dim), name='input_X1')
    hidden = Dense(100, activation='relu')(input_X1)
    hidden = Dense(200, activation='relu')(hidden)
    hidden = Dense(200, activation='relu')(hidden)
    hidden = Dense(100, activation='relu')(hidden)
    hidden = Dense(50, activation='relu')(hidden)
    hidden = Dense(50, activation='relu')(hidden)
    hidden = Dense(30, activation='relu')(hidden)
    hidden = Dense(30, activation='relu')(hidden)
    hidden = Dense(30, activation='relu')(hidden)
    hidden = Dense(10, activation='relu')(hidden)
    z1 = Dense(8, activation='relu')(hidden)

    # TODO: デコーダ用の層を作成する

    # 単一モダリティでの分類用のネットワーク
    c_x1 = Dropout(0.5)(hidden)
    c_x1 = Dense(8, activation='softmax')(c_x1)
    x1_single_model = Model(input_X1, c_x1)
    
    return input_X1, z1, x1_single_model

def X2_encoder(X2_dim):
    # モダリティ2の特徴量抽出層
    input_X2 = Input(shape=(None, X2_dim), name='input_X2')
    hidden = Dense(1000, activation='relu')(input_X2)
    hidden = Dense(2000, activation='relu')(hidden)
    hidden = Dense(2000, activation='relu')(hidden)
    hidden = Dense(1000, activation='relu')(hidden)
    hidden = Dense(500, activation='relu')(hidden)
    hidden = Dense(500, activation='relu')(hidden)
    hidden = Dense(100, activation='relu')(hidden)
    hidden = Dense(100, activation='relu')(hidden)
    hidden = Dense(100, activation='relu')(hidden)
    hidden = Dense(50, activation='relu')(hidden)
    hidden = Dense(10, activation='relu')(hidden)
    z2 = Dense(8, activation='relu')(hidden)

    # TODO: デコーダ用の層を作成する

    # 単一モダリティでの分類用のネットワーク
    c_x2 = Dropout(0.5)(hidden)
    c_x2 = Dense(8, activation='softmax')(c_x2)
    x2_single_model = Model(input_X2, c_x2)


    return input_X2, z2, x2_single_model

# デコーダ
def X1_decoder(X1_dim):
    # モダリティ1の復元
    dec_input_X1 = Input(batch_shape=(None, X1_dim))
    hidden = Dense(16, activation='relu')(dec_input_X1)
    hidden = Dense(32, activation='relu')(hidden)
    hidden = Dense(64, activation='relu')(hidden)
    hidden = Dense(128, activation='relu')(hidden)
    hidden = Dense(256, activation='relu')(hidden)
    dec_output_X1 = Dense(533, activation='relu')(hidden)

    dec_X1 = Model(dec_input_X1, dec_output_X1)

    return dec_X1

def X2_decoder(X2_dim):
    # モダリティ2の復元
    dec_input_X2 = Input(batch_shape=(None, X2_dim))
    hidden = Dense(16, activation='relu')(dec_input_X2)
    hidden = Dense(32, activation='relu')(hidden)
    hidden = Dense(64, activation='relu')(hidden)
    dec_output_X2 = Dense(128, activation='relu')(hidden)

    dec_X2 = Model(dec_input_X2, dec_output_X2)

    return dec_X2

# 分類層
def classification_layer(input_X1, input_X2, z1, z2):
    # 特徴量を合成
    # TODO: MaxPooling実装したい

    concat = Concatenate()([z1, z2])

    # 分類層
    classification_input = Dense(8, activation='relu', name='classification_1')(concat)     # concat or maxpooling
    classification = Dense(8, activation='relu', name='classification_2')(classification_input)
    #classification = Dense(8, activation='relu', name='classification_7')(classification)

    # 出力層
    classification_output = Dropout(0.5)(classification)
    output = Dense(8, activation='softmax', name='output_layer')(classification_output)

    multimodal_model = Model([input_X1, input_X2], output)

    return multimodal_model

# 教師あり学習
def supervised_learning(X1, X2, y):      # セットになったデータのみ学習
    # データを分割
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, shuffle=True, test_size=0.2, random_state=0)
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1_train, X2_train, y_train, shuffle=True, test_size=0.2, random_state=0)

    # モデルを定義
    # 各種パラメータを決定
    length = len(X1_train)          # 学習データの数
    X1_dim = X1_train.shape[1]      # モダリティ1(音声)の次元数
    X2_dim = X2_train.shape[1]      # モダリティ2(テキスト)の次元数

    # DEBUG
    print("X1_train.shape:", X1_train.shape)        # 学習用モダリティ1(データ数, 入力次元数)
    print("X2_train.shape:", X2_train.shape)        # 学習用モダリティ2(データ数, 入力次元数)
    print("y_train.shape:", y_train.shape)          # 学習用ラベル(データ数, クラス数)

    # 特徴量抽出層
    input_X1, z1, x1_single_model = X1_encoder(X1_dim)
    input_X2, z2, x2_single_model = X2_encoder(X2_dim)

    # 分類層
    multimodal_model = classification_layer(input_X1, input_X2, z1, z2)

    # モデル生成
    multimodal_model.compile(optimizer=Adam(lr=1e-4, decay=1e-6, amsgrad=True), loss=categorical_crossentropy, metrics=['accuracy'])

    x1_single_model.compile(optimizer=Adam(lr=1e-4, decay=1e-6, amsgrad=True), loss=categorical_crossentropy, metrics=['accuracy'])

    x2_single_model.compile(optimizer=Adam(lr=1e-4, decay=1e-6, amsgrad=True), loss=categorical_crossentropy, metrics=['accuracy'])

    
    # ----------------------------------------------------------------------------------------
    # モデルの学習
    epochs = 300        # 学習用パラメータ
    batch_size = 64

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
    print("classification report multimodal\n",classification_report(result_multimodal_test, result_multimodal_pred))
    print("classification report x1\n", classification_report(result_x1_test, result_x1_pred))
    print("classification report x2\n", classification_report(result_x2_test, result_x2_pred))

    # 結果を保存
    supervised_train_save_log(multimodal_model, x1_single_model, x2_single_model, multimodal_fit, x1_fit, x2_fit)

# 半教師あり学習
def semi_supervised_learning(X1, X2, un_X1, un_X2, y):          # すべてのデータで学習
    # データを分割
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, shuffle=True, test_size=0.2, random_state=0)
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1_train, X2_train, y_train, shuffle=True, test_size=0.2, random_state=0)


    # モデルを定義
    # 各種パラメータを決定
    length = len(X1_train)          # 学習データの数
    X1_dim = X1_train.shape[1]      # モダリティ1(音声)の次元数
    X2_dim = X2_train.shape[1]      # モダリティ2(テキスト)の次元数

    # DEBUG
    print("X1_train.shape:", X1_train.shape)        # 学習用モダリティ1(データ数, 入力次元数)
    print("X2_train.shape:", X2_train.shape)        # 学習用モダリティ2(データ数, 入力次元数)
    print("y_train.shape:", y_train.shape)          # 学習用ラベル(データ数, クラス数)

    # 特徴量抽出層
    input_X1, z1 = X1_encoder(X1_dim)
    input_X2, z2 = X2_encoder(X2_dim)

    # 分類層
    multimodal_model = classification_layer(input_X1, input_X2, z1, z2)

    # モデル生成
    multimodal_model.compile(optimizer=Adam(lr=1e-4, decay=1e-6, amsgrad=True), loss=categorical_crossentropy, metrics=['accuracy'])


# 教師あり学習のログを保存
def supervised_train_save_log(multimodal_model, x1_single_model, x2_single_model, multimodal_fit, x1_fit, x2_fit):
    now = datetime.datetime.now()                                   # 現在時刻を取得
    file_name = now.strftime('%Y%m%d_%H%M')                         # 現在時刻を文字列として格納

    make_dir = "./train_log/" +  file_name
    os.mkdir(make_dir)                                              # 現在時刻のディレクトリを作成

    # モデルの構成を保存(.png)
    plot_model(multimodal_model, to_file=make_dir + '/model_shape_MM' + file_name + '.png', show_shapes=True)
    plot_model(x1_single_model, to_file=make_dir + '/model_shape_x1' + file_name + '.png', show_shapes=True)
    plot_model(x2_single_model, to_file=make_dir + '/model_shape_x2' + file_name + '.png', show_shapes=True)

    # 学習ログ(.csv)
    df1 = pd.DataFrame(multimodal_fit.history)      # DataFrame化
    df2 = pd.DataFrame(x1_fit.history)
    df3 = pd.DataFrame(x2_fit.history)

    df1.to_csv(make_dir + '/train_log_MM' + file_name + '.csv')     # csvで保存
    df2.to_csv(make_dir + '/train_log_x1' + file_name + '.csv')
    df3.to_csv(make_dir + '/train_log_x2' + file_name + '.csv')

    # グラフ(.png)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(2, 1, 1)      # multimodal loss
    ax2 = fig1.add_subplot(2, 1, 2)      # multimodal acc

    ax1.plot(multimodal_fit.history['loss'])
    ax1.plot(multimodal_fit.history['val_loss'])
    ax1.set_title('multimodal loss')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    

    ax2.plot(multimodal_fit.history['accuracy'])
    ax2.plot(multimodal_fit.history['val_accuracy'])
    ax2.set_title('multimodal accuracy')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(make_dir + "/reslt_multimodal_graph" + file_name + '.png')

    fig2 = plt.figure()
    ax3 = fig2.add_subplot(2, 1, 1)      # x1 loss
    ax4 = fig2.add_subplot(2, 1, 2)      # x1 acc

    ax3.plot(x1_fit.history['loss'])
    ax3.plot(x1_fit.history['val_loss'])
    ax3.set_title('x1 loss')
    ax3.legend(['Train', 'Validation'], loc='upper left')

    ax4.plot(x1_fit.history['accuracy'])
    ax4.plot(x1_fit.history['val_accuracy'])
    ax4.set_title('x1 accuracy')
    ax4.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(make_dir + "/reslt_x1_graph" + file_name + '.png')

    fig3 = plt.figure()
    ax5 = fig3.add_subplot(2, 1, 1)      # x2 loss 
    ax6 = fig3.add_subplot(2, 1, 2)      # x2 acc

    ax5.plot(x2_fit.history['loss'])
    ax5.plot(x2_fit.history['val_loss'])
    ax5.set_title('x2 loss')
    ax5.legend(['Train', 'Validation'], loc='upper left')

    ax6.plot(x2_fit.history['accuracy'])
    ax6.plot(x2_fit.history['val_accuracy'])
    ax6.set_title('x2 accuracy')
    ax6.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(make_dir + "/reslt_x2_graph" + file_name + '.png')

    plt.show()

def main():
    # メタデータのディレクトリ
    meta_data = pd.read_csv("data/OGVC_Vol1_supervised.csv", header=0)
    supervised_meta = meta_data.dropna(subset=['emotion'], axis=0)      # 全体のメタデータから教師ありデータのみを抽出

    # ラベルの読み込み
    # ワンホットエンコーディング
    ohe = OneHotEncoder(sparse=False)
    encoded = ohe.fit_transform(supervised_meta[['emotion']].values)

    label = ohe.get_feature_names(['emotion'])
    label_list = pd.DataFrame(encoded, columns=label, dtype=np.int8)
    
    # 教師ありデータの読み込み
    sound_labeled_X1 = pd.read_csv("train_data/2div/POW_labeled.csv", header=0, index_col=0)
    tfidf_labeled_X2 = pd.read_csv("train_data/2div/TF-IDF_labeled_PCA.csv", header=0, index_col=0)

    X1 = sound_labeled_X1.to_numpy()        # 学習データをnumpy配列に変換
    X2 = tfidf_labeled_X2.to_numpy()
    y = label_list.to_numpy()

    print(X1)


    # データを標準化
    x_mean = X1.mean(keepdims=True)
    x_std = X1.std(keepdims=True)
    #X1 = (X1 - x_mean)/(x_std)

    x_min = X2.min(keepdims=True)
    x_max = X2.max(keepdims=True)
    #X2 = (X2 - x_min)/(x_max - x_min) 

    print(X1)

    # モードを選択
    print("\n--\nSelect a function to execute")
    print("[0]supervised_learning\n[1]semi_supervised_learning\n")
    mode = input("imput the number:")

    # 実行する関数によって分岐
    if mode == '0':         # 教師ありデータのみで学習
        # 教師ありデータとラベルを表示
        print("\n\nsupervised sound data\n", sound_labeled_X1.head(), "\n")
        print("supervised tfidf data\n", tfidf_labeled_X2.head(), "\n")
        print("label data\n", label_list.head(), "\n")

        # 教師あり学習を実行
        supervised_learning(X1, X2, y)

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

        un_X1 = sound_un_labeled_X1.to_numpy()        # 学習データをnumpy配列に変換
        un_X2 = tfidf_un_labeled_X2.to_numpy()        # 学習データをnumpy配列に変換

        semi_supervised_learning(X1, X2, un_X1, un_X2, y)

    else:
        print("error")

if __name__ == "__main__":
    main()