# Created by RitsukiShuto on 2022/08/01.
# train_model.py
#
import datetime
import os
import random
from re import split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import Input, Model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import (Add, Concatenate, Conv1D, Dense, Dropout, Flatten, MaxPool1D)
from keras.losses import categorical_crossentropy
from keras.utils.vis_utils import plot_model

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

now = datetime.datetime.now()       # 現在時刻を取得

# エンコーダ
def X1_encoder(X1_dim):
    # モダリティ1の特徴量抽出層
    input_X1 = Input(shape=(X1_dim, 1), name="input_X1")
    #input_X1 = Input(batch_shape=(None, X1_dim), name='input_X1_DNN')

    hidden = Dense(10, activation='relu')(input_X1)
    hidden = Dense(10, activation='relu')(hidden)
    hidden = Dense(10, activation='relu')(hidden)
    #hidden = Dense(8, activation='relu')(hidden)

    #z1 = Dense(10, activation='relu')(hidden)

    hidden = Conv1D(10, 2, padding='same', activation='relu')(hidden)
    hidden = MaxPool1D(pool_size=2, padding='same')(hidden)

    z1 = Flatten()(hidden)

    # 単一モダリティでの分類用のネットワーク
    c_x1 = Dropout(0.5)(z1)
    c_x1 = Dense(5, activation='softmax')(c_x1)
    x1_single_model = Model(input_X1, c_x1)
    
    return input_X1, z1, x1_single_model

def X2_encoder(X2_dim):
    # モダリティ2の特徴量抽出層
    input_X2 = Input(shape=(X2_dim, 1), name='input_X2')
    #input_X2 = Input(batch_shape=(None, X2_dim), name='input_X2_DNN')

    hidden = Dense(300, activation='relu')(input_X2)
    hidden = Dense(200, activation='relu')(hidden)
    hidden = Dense(100, activation='relu')(hidden)
    hidden = Dense(50, activation='relu')(hidden)
    hidden = Dense(25, activation='relu')(hidden)
    hidden = Dense(25, activation='relu')(hidden)

    #z2 = Dense(10, activation='relu')(hidden)

    hidden = Conv1D(10, 2, padding='same', activation='relu')(hidden)
    hidden = MaxPool1D(pool_size=2, padding='same')(hidden)

    z2 = Flatten()(hidden)

    # TODO: デコーダ用の層を作成する

    # 単一モダリティでの分類用のネットワーク
    c_x2 = Dropout(0.5)(z2)
    c_x2 = Dense(5, activation='softmax')(c_x2)
    x2_single_model = Model(input_X2, c_x2)

    return input_X2, z2, x2_single_model

# 分類層
def classification_layer(input_X1, input_X2, z1, z2):
    # 特徴量を合成
    # TODO: MaxPooling実装したい
    concat = Concatenate()([z1, z2])

    # 分類層
    classification = Dense(20, activation='relu', name='classification_1')(concat)

    classification = Dense(15, activation='relu', name='classification_2')(classification)
    classification = Dense(15, activation='relu', name='classification_3')(classification)
    classification = Dense(10, activation='relu', name='classification_4')(classification)

    classification = Dense(10, activation='relu', name='classification_5')(classification)
    #classification = MaxPool1D(pool_size=4, padding='same')(classification)

    #classification = Flatten()(classification)

    # 出力層
    classification_output = Dropout(0.5)(classification)
    output = Dense(5, activation='softmax', name='output_layer')(classification_output)

    multimodal_model = Model([input_X1, input_X2], output)

    return multimodal_model

# 学習
def model_fit(X1_train, X2_train, y_train, X1_test, X2_test, y_test, meta_data):
# 各種パラメータを決定
    length = len(X1_train)          # 学習データの数
    X1_dim = X1_train.shape[1]      # モダリティ1(音声)の次元数
    X2_dim = X2_train.shape[1]      # モダリティ2(テキスト)の次元数

    # 特徴量抽出層
    input_X1, z1, x1_single_model = X1_encoder(X1_dim)
    input_X2, z2, x2_single_model = X2_encoder(X2_dim)

    # 分類層
    multimodal_model = classification_layer(input_X1, input_X2, z1, z2)

    # モデル生成
    multimodal_model.compile(optimizer=Adam(lr=1e-4, decay=1e-6, amsgrad=True), loss=categorical_crossentropy, metrics=['accuracy'])

    x1_single_model.compile(optimizer=Adam(lr=1e-4, decay=1e-6, amsgrad=True), loss=categorical_crossentropy, metrics=['accuracy'])

    x2_single_model.compile(optimizer=Adam(lr=1e-4, decay=1e-6, amsgrad=True), loss=categorical_crossentropy, metrics=['accuracy'])

    # epochsとbatch_size
    epochs = 500        # 学習用パラメータ(デフォルトはe=250, b=64)
    batch_size = 256

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10)

    multimodal_fit = multimodal_model.fit(x=[X1_train, X2_train],
                                          y=y_train,
                                          validation_split=0.2,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          callbacks=[early_stopping],
                                          verbose=0  # type: ignore
                                          )

    x1_fit = x1_single_model.fit(x=X1_train,
                                 y=y_train,
                                 validation_split=0.2,
                                 batch_size = batch_size,
                                 epochs=epochs,
                                 callbacks=[early_stopping],
                                 verbose=0  # type: ignore
                                 )

    x2_fit = x2_single_model.fit(x=X2_train,
                                 y=y_train,
                                 validation_split=0.2,
                                 batch_size = batch_size,
                                 epochs=epochs,
                                 callbacks=[early_stopping],
                                 verbose=0  # type: ignore
                                 )

    # モデルを保存
    MM_model = "models/multimodal/multimodal_model" + now.strftime('%Y%m%d_%H%M')       # ファイル名を生成
    x1_model = "models/x1/x1_model" + now.strftime('%Y%m%d_%H%M')
    x2_model = "models/x2/x2_model" + now.strftime('%Y%m%d_%H%M')

    # 学習済みモデルを保存
    multimodal_model.save(MM_model)
    x1_single_model.save(x1_model)
    x2_single_model.save(x2_model)

    # モデルを評価
    MM_confusion_matrix = evaluate_model(multimodal_model, x1_single_model, x2_single_model, X1_test, X2_test, y_test, meta_data)

    return multimodal_model, x1_single_model, x2_single_model, multimodal_fit, x1_fit, x2_fit, MM_confusion_matrix

# モデルの評価
def evaluate_model(multimodal_model, x1_single_model, x2_single_model, X1_test, X2_test, y_test, meta_data):

    X1_df = pd.read_csv("train_data/OGVC_vol1/POW_labeled.csv", header=0)
    X1 = X1_df.values.tolist()

    # テストデータで推定する
    pred_MM = multimodal_model.predict(x=[X1_test, X2_test])
    pred_X1 = x1_single_model.predict(X1_test)
    pred_X2 = x2_single_model.predict(X2_test)

    MM_pred_ = np.argmax(pred_MM, axis=1)
    X1_pred_ = np.argmax(pred_X1, axis=1)
    X2_pred_ = np.argmax(pred_X2, axis=1)

    y_test_ = np.argmax(y_test, axis=1)

    # クラスごとの分類精度を表示する。
    MM_confusion_matrix = confusion_matrix(y_test_, MM_pred_)
    X1_confusion_matrix = confusion_matrix(y_test_, X1_pred_)
    X2_confusion_matrix = confusion_matrix(y_test_, X2_pred_)

    # 精度を表示
    MM_score = multimodal_model.evaluate(x=[X1_test, X2_test], y=y_test, verbose=0)
    X1_score = x1_single_model.evaluate(X1_test, y_test, verbose=0)
    X2_score = x2_single_model.evaluate(X2_test, y_test, verbose=0)

    #multimodal_model.summary()
    print("Multimodal score")
    print("Test loss:", MM_score[0])
    print("test accuracy:", MM_score[1], "\n")
    #print("confusion matrix", MM_confusion_matrix)

    #x1_single_model.summary()
    print("X1 score")
    print("Test loss:", X1_score[0])
    print("test accuracy:", X1_score[1], "\n")
    #print("confusion matrix", X1_confusion_matrix)

    #x2_single_model.summary()
    print("X2 score")
    print("Test loss:", X2_score[0])
    print("test accuracy:", X2_score[1], "\n")
    #print("confusion matrix", X2_confusion_matrix)

    # 不正解ログ作成の準備
    incorrect_ans_list = []
    correct_ans_list = []
    label = ['ACC', 'ANG', 'ANT', 'DIS', 'FEA', 'JOY', 'SAD', 'SUR']

    # 不正解のデータを抽出
    for i,v in enumerate(pred_MM):      # multimodal
        pre_ans = v.argmax()
        ans = y_test[i].argmax()

        # BUG: ログが保存されない
        # インデックスがおかしいかも
        if ans != pre_ans:      # 不正解
            # メタデータから不正解のデータを探す
            for j in range(len(X1)):
                if list(X1_test[i][0:]) == list(X1[j][1:]):
                    # ラベルを数値から文字列に変更
                    pre_label = label[pre_ans]
                    ans_label = label[ans]

                    # ファイル名と連番を取得
                    name = X1[j][0]
                    idx = str.rfind(name, "_")
                    f_name = name[:idx]             # ファイル名
                    number = name[idx+1:]           # ファイル番号

                    # メタデータから不正解の発話文字列を探す
                    for row in meta_data.values:
                        if f_name == row[0] and int(number) == row[1]:
                            # 不正解のリストを保存
                            incorrect_meta = [f_name, number, pre_label, row[6], row[7], row[8], ans_label, row[5]]
                            incorrect_ans_list.append(incorrect_meta)

        else:                   # 正解
            # メタデータから不正解のデータを探す
            for j in range(len(X1)):
                if list(X1_test[i][0:]) == list(X1[j][1:]):
                    # ラベルを数値から文字列に変更
                    pre_label = label[pre_ans]
                    ans_label = label[ans]

                    # ファイル名と連番を取得
                    name = X1[j][0]
                    idx = str.rfind(name, "_")
                    f_name = name[:idx]             # ファイル名
                    number = name[idx+1:]           # ファイル番号

                    # メタデータから不正解の発話文字列を探す
                    for row in meta_data.values:
                        if f_name == row[0] and int(number) == row[1]:
                            #不正解のリストを保存
                            correct_meta = [f_name, number, pre_label, row[6], row[7], row[8], ans_label, row[5]]
                            correct_ans_list.append(correct_meta)

    # FIXME: OGVC_vol.2に対応する
    # 不正解データの保存
    df1 = pd.DataFrame(incorrect_ans_list, columns = ['file name', 'f_num', 'pred', 'ans1', 'ans2', 'ans3', 'emotion', 'text'])
    df1 = df1.sort_values(by=["file name", "f_num"])
    df1.to_csv("predict_log/incorrect_ans_list_" + now.strftime('%Y%m%d_%H%M') + ".csv")

    # 正解データの保存
    df2 = pd.DataFrame(correct_ans_list, columns = ['file name', 'f_num', 'pred', 'ans1', 'ans2', 'ans3', 'emotion', 'text'])
    df2 = df2.sort_values(by=["file name", "f_num"])
    df2.to_csv("predict_log/correct_ans_list_" + now.strftime('%Y%m%d_%H%M') + ".csv")

    return MM_confusion_matrix

# 混同行列の整形, 保存
def calc_conf_mat(MM_confusion_matrix):
    df_conf_mat = pd.DataFrame(MM_confusion_matrix, columns=['ANG', 'JOY', 'NEU', 'SAD', 'SUR'])
    df_conf_mat['Number of test data by class'] = df_conf_mat.sum(axis=1)       # クラスごとの合計を計算
    df_conf_mat.at['5', 'Number of test data by class'] = df_conf_mat['Number of test data by class'].sum(axis=0)       # テストデータの合計を計算

    df_conf_mat_prob = np.empty((5, 5))

    for j in range(5):
        for k in range(5):
            # 各要素ごとの確率を計算して格納
            df_conf_mat_prob[j, k] = df_conf_mat.iloc[j, k] / df_conf_mat.at[j, 'Number of test data by class']

        # DataFrameに変換
        df_conf_mat_prob = pd.DataFrame(df_conf_mat_prob,
                                        columns=['ANG', 'JOY','NEU', 'SAD', 'SUR'],
                                        index=['ANG', 'JOY', 'NEU', 'SAD', 'SUR'])

        # 精度を格納
        df_conf_mat_prob.at['5', 'accuracy'] = np.sum(np.diag(df_conf_mat_prob)) / df_conf_mat.at['5', 'Number of test data by class']

        # 保存
        now = datetime.datetime.now()       # 現在時刻を取得
        now_str = now.strftime('%Y%m%d_%H%M%S')
        df_conf_mat.to_csv("conf_mat/multimodal/confusion_matrix" + now_str + ".csv", sep=',')
        df_conf_mat_prob.to_csv("conf_mat/multimodal/confusion_matrix_prob" + now_str + ".csv", sep=',')

# ログを保存
def save_log(multimodal_model, x1_single_model, x2_single_model, multimodal_fit, x1_fit, x2_fit):
    # ログを保存
    file_name = now.strftime('%Y%m%d_%H%M')                         # 現在時刻を文字列として格納

    make_dir = "./train_log/" +  file_name
    os.mkdir(make_dir)                                              # 現在時刻のディレクトリを作成

    # ネットワークの構成を保存(.png)
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
    ax1.set_title('multimodal loss')
    ax1.legend(['Train'], loc='upper left')
    
    ax2.plot(multimodal_fit.history['accuracy'])
    ax2.set_title('multimodal accuracy')
    ax2.legend(['Train'], loc='upper left')

    plt.savefig(make_dir + "/reslt_multimodal_graph" + file_name + '.png')

    fig2 = plt.figure()
    ax3 = fig2.add_subplot(2, 1, 1)      # x1 loss
    ax4 = fig2.add_subplot(2, 1, 2)      # x1 acc

    ax3.plot(x1_fit.history['loss'])
    ax3.set_title('x1 loss')
    ax3.legend(['Train'], loc='upper left')

    ax4.plot(x1_fit.history['accuracy'])
    ax4.set_title('x1 accuracy')
    ax4.legend(['Train'], loc='upper left')

    plt.savefig(make_dir + "/reslt_x1_graph" + file_name + '.png')

    fig3 = plt.figure()
    ax5 = fig3.add_subplot(2, 1, 1)      # x2 loss 
    ax6 = fig3.add_subplot(2, 1, 2)      # x2 acc

    ax5.plot(x2_fit.history['loss'])
    ax5.set_title('x2 loss')
    ax5.legend(['Train'], loc='upper left')

    ax6.plot(x2_fit.history['accuracy'])
    ax6.set_title('x2 accuracy')
    ax6.legend(['Train'], loc='upper left')

    plt.savefig(make_dir + "/reslt_x2_graph" + file_name + '.png')
    plt.show()

# 教師あり学習
def supervised_learning(X1_train, X1_test, X2_train, X2_test, y_train, y_test, meta_data):      # セットになったデータのみ学習
    result = np.empty((0, 5, 5))

    # 半教師あり学習と同数のデータで学習
    #X1_sv, X1_un, X2_sv, X2_un, y_sv, y_un = train_test_split(X1_train, X2_train, y_train, shuffle=True, test_size=0.7, random_state=0, stratify=y_train)

    #print("学習データ件数:", X1_sv.shape[0])  # type: ignore
    print("テストデータ件数:", X1_test.shape[0])

    label_cnt = pd.DataFrame(y_train, columns=['ANG', 'JOY', 'NEU', 'SAD', 'SUR'])

    print("\n学習データのクラスごとの件数")
    print("ANG", len(label_cnt.query('ANG == 1')))
    print("JOY", len(label_cnt.query('JOY == 1')))
    print("NEU", len(label_cnt.query('NEU == 1')))
    print("SAD", len(label_cnt.query('SAD == 1')))
    print("SUR", len(label_cnt.query('SUR == 1')))

    print("学習データ総計:", len(label_cnt))

    for i in range(10):
        print("\nループ回数:", i+1, "\n")

        # モデルを学習
        multimodal_model, x1_single_model, x2_single_model, multimodal_fit, x1_fit, x2_fit, MM_confusion_matrix = model_fit(X1_train, X2_train, y_train, X1_test, X2_test, y_test, meta_data)

        calc_conf_mat(MM_confusion_matrix)

        # 混同行列を格納
        conf_mat = np.reshape(MM_confusion_matrix, (1, 5, 5))
        result = np.append(result, conf_mat, axis=0)

    print(result)       # DEBUG

    # 平均と分散を計算
    avg_conf_mat = np.average(result, axis=0)
    var_conf_mat = np.var(result, axis=0)

    # 配列をDataFrameに変換
    df_avg = pd.DataFrame(avg_conf_mat, columns=['ANG', 'JOY', 'NEU', 'SAD', 'SUR'])
    df_var = pd.DataFrame(var_conf_mat, columns=['ANG', 'JOY', 'NEU', 'SAD', 'SUR'])

    #calc_conf_mat(df_avg)

    print("\n", df_avg, "\n")
    print(df_var)

    # ログを保存
    save_log(multimodal_model, x1_single_model, x2_single_model, multimodal_fit, x1_fit, x2_fit)

# 半教師あり学習
def semi_supervised_learning(X1_train, X1_sv, X1_un, X1_test, X2_train, X2_sv, X2_un, X2_test, y_train, y_sv, y_test, meta_data):          # すべてのデータで学習
    print("semi supervised learning")       # DEBUG

    label_cnt = pd.DataFrame(y_sv, columns=['ANG', 'JOY', 'NEU', 'SAD', 'SUR'])

    print(f"教師ありデータ:{X1_sv.shape[0]}\n教師なしデータ:{X1_un.shape[0]}")

    print("\n学習データのクラスごとの件数")
    print("ANG", len(label_cnt.query('ANG == 1')))
    print("JOY", len(label_cnt.query('JOY == 1')))
    print("NEU", len(label_cnt.query('NEU == 1')))
    print("SAD", len(label_cnt.query('SAD == 1')))
    print("SUR", len(label_cnt.query('SUR == 1')))

    # ラベルなしデータを読み込む
    sound_un_labeled_X1 = pd.read_csv("train_data/mixed/MFCC_un_labeled.csv", header=0, index_col=0)
    tfidf_un_labeled_X2 = pd.read_csv("train_data/mixed/TF-IDF_un_labeled.csv", header=0, index_col=0)

    # データを変換
    un_X1 = sound_un_labeled_X1.to_numpy()        # 学習データをnumpy配列に変換
    un_X2 = tfidf_un_labeled_X2.to_numpy()        # 学習データをnumpy配列に変換

    # 新規読み込みデータと教師ありデータを結合
    un_X1 = np.append(X1_un, un_X1, axis=0)
    un_X2 = np.append(X2_un, un_X2, axis=0)

    # ループ回数等に関わる変数
    data_cnt = un_X1.shape[0]   # データ件数
    ref_dara_range = 100
    loop_times = data_cnt / ref_dara_range      # ループ回数
    last_loop = data_cnt - ref_dara_range       # TODO: ラベルなしデータの端数部を処理するための変数

    # 推定時のパラメータ
    batchsize = 256
    reliableness = 0.3

    for i in range(10):

        print("\nループ回数:", i+1, "\n")
        result = []

        # 初回学習
        print("###初回学習を開始###")
        multimodal_model, x1_single_model, x2_single_model, multimodal_fit, x1_fit, x2_fit, MM_confusion_matrix = model_fit(X1_sv, X2_sv, y_sv, X1_test, X2_test, y_test, meta_data)
        calc_conf_mat(MM_confusion_matrix)

        # 未ラベルデータの参照範囲
        start = 0
        end = ref_dara_range

        # 疑似ラベルの生成
        for j in range(int(loop_times)):
            print(j+1, "/", int(loop_times))
            print(start, "to", end)

            # ラベルなしデータを推定
            MM_encoded = multimodal_model.predict(x=[un_X1[start:end][0:], un_X2[start:end][0:]], batch_size=batchsize)

            # 信頼度が高い順に20件のデータをピックアップ
            top20_index = np.argpartition(np.max(MM_encoded, axis=1), -20)[-20:]
            temp_label = np.zeros((20, 5), dtype=int)

            for l in range(len(top20_index)):
                temp_label[l][np.argmax(MM_encoded[top20_index[l]])] = 1
 
            X1_train = np.append(X1_train, un_X1[top20_index + start], axis=0)      # 教師ありデータにスタック
            X2_train = np.append(X2_train, un_X2[top20_index + start], axis=0)
            y_train = np.append(y_train, temp_label,axis=0)

            print(top20_index + start)

            # データをシャッフル
            np.random.seed(0)               # ランダムシードを固定
            np.random.shuffle(X1_train)     # シャッフル

            np.random.seed(0)
            np.random.shuffle(X2_train)

            np.random.seed(0)
            np.random.shuffle(y_train)

            print("追加データ数:", len(top20_index))
            multimodal_model, x1_single_model, x2_single_model, multimodal_fit, x1_fit, x2_fit, MM_confusion_matrix = model_fit(X1_train, X2_train, y_train, X1_test, X2_test, y_test, meta_data)

            #calc_conf_mat(MM_confusion_matrix)

            start = end + 1
            end += ref_dara_range

        # 未知のデータでテスト
        MM_confusion_matrix = evaluate_model(multimodal_model, x1_single_model, x2_single_model, X1_test, X2_test, y_test, meta_data)

        # 混同行列を格納
        MM_confusion_matrix = np.reshape(MM_confusion_matrix, (1, 5, 5))
        result = np.append(result, MM_confusion_matrix, axis=0)

        print(f"{i}/10 reslut")
        print(result)

    avg_conf_mat = np.average(result, axis=0)
    
    #var_conf_mat = np.var(result, axis=0)

    # 配列をDataFrameに変換
    df_avg = pd.DataFrame(avg_conf_mat, columns=['ANG', 'JOY', 'NEU', 'SAD', 'SUR'])
    #df_var = pd.DataFrame(var_conf_mat, columns=['ANG', 'JOY', 'NEU', 'SAD', 'SUR'])

    calc_conf_mat(df_avg)

    print(df_avg)
    #print(df_var)

    save_log(multimodal_model, x1_single_model, x2_single_model, multimodal_fit, x1_fit, x2_fit)

def main():
    # メタデータのディレクトリ
    # CAUTION: 使用するメタデータを変更する
    meta_data = pd.read_csv("train_data/meta_data/MOY_mixed_meta.csv", header=0)  # INFO: OGVC_vol.1
    #meta_data = pd.read_csv("data/OGVC_Vol2_supervised.csv", header=0)  # INFO: OGVC_vol.2
    supervised_meta = meta_data.dropna(subset=['emotion'], axis=0)      # 全体のメタデータから教師ありデータのみを抽出

    # ラベルの読み込み
    # ワンホットエンコーディング
    ohe = OneHotEncoder(sparse=False)
    encoded = ohe.fit_transform(supervised_meta[['emotion']].values)

    label = ohe.get_feature_names(['emotion'])
    label_list = pd.DataFrame(encoded, columns=label, dtype=np.int8)
    
    # 教師ありデータの読み込み
    # INFO: OGVC_vol.1
    sound_labeled_X1 = pd.read_csv("train_data/mixed/MFCC_labeled.csv", header=0, index_col=0)
    tfidf_labeled_X2 = pd.read_csv("train_data/mixed/TF-IDF_labeled.csv", header=0, index_col=0)

    # INFO OGVC_vol.2
    #sound_labeled_X1 = pd.read_csv("train_data/OGVC_vol2/POW_all.csv", header=0, index_col=0)
    #tfidf_labeled_X2 = pd.read_csv("train_data/OGVC_vol2/TF-IDF_labeled_PCA.csv", header=0, index_col=0)

    # 学習データをnumpy配列に変換
    X1 = sound_labeled_X1.to_numpy()
    X2 = tfidf_labeled_X2.to_numpy()
    y = label_list.to_numpy()

    # データを分割
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, shuffle=True, test_size=0.2, random_state=0, stratify=y)

    # モードを選択
    print("\n--\nSelect a function to execute")
    print("[0]supervised_learning\n[1]semi_supervised_learning\n[2]evaluate_model\n")
    mode = input("imput the number:")

    # 実行する関数によって分岐
    if mode == '0':         # 教師ありマルチモーダル学習        
        # 教師ありデータとラベルを表示
        print("\n\nsupervised sound data\n", sound_labeled_X1.head(), "\n")
        print("supervised tfidf data\n", tfidf_labeled_X2.head(), "\n")
        print("label data\n", label_list.head(), "\n")

        # 教師あり学習を実行
        supervised_learning(X1_train, X1_test, X2_train, X2_test, y_train, y_test, supervised_meta)

    elif mode == '1':       # 半教師ありマルチモーダル学習
        # TODO: モデルの読み込みとデータ分割の関数を作ってもいいかも
        X1_sv, X1_un, X2_sv, X2_un, y_sv, y_un = train_test_split(X1_train, X2_train, y_train, shuffle=True, test_size=0.8, random_state=0, stratify=y_train)

        #semi_supervised_learning(multimodal_model, X1_train, X1_test, X2_train, X2_test, y_train, y_test, meta_data)
        semi_supervised_learning(X1_train, X1_sv, X1_un, X1_test, X2_train, X2_sv, X2_un, X2_test, y_train, y_sv, y_test, meta_data)

    elif mode == "2":
        # モデルを読み込む
        # TODO: 読み込むモデルを選べるようにする
        multimodal_model = tf.keras.models.load_model("models/multimodal/multimodal_model20221015_1246")
        x1_single_model =  tf.keras.models.load_model("models/x1/x1_model20221015_1246")
        x2_single_model =  tf.keras.models.load_model("models/x2/x2_model20221015_1246")

        # データを分割
        X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, shuffle=True, test_size=0.2, random_state=0)

        evaluate_model(multimodal_model, x1_single_model, x2_single_model,
                        X1_test, X2_test, y_test, meta_data)

    else:
        print("error")

if __name__ == "__main__":
    main()
