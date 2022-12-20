# Created by RitsukiShuto on 2022/11/23.
# main.py
# ニューラルネットワークの定義, 学習を行う
#
import tensorflow as tf
from keras import Input, Model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import (Add, Concatenate, Conv1D, Dense, Dropout, Flatten, MaxPool1D)
from keras.losses import categorical_crossentropy
from keras.utils.vis_utils import plot_model

# 学習
def model_fit(X_train, Y_train, Z_train, epochs):
    # モダリティの次元数を取得
    X_dim = X_train.shape[1]
    Y_dim = Y_train.shape[1]

    # 特徴量抽出層
    X_input, X_feature, X_single_model = X_encoder(X_dim)
    Y_input, Y_feature, Y_single_model = Y_encoder(Y_dim)

    # マルチモーダル分類層
    multimodal_model = Multimodal_Classification_Layer(X_input, Y_input, X_feature, Y_feature)

    # モデルを生成
    multimodal_model.compile(optimizer=Adam(lr=1e-4, decay=1e-6, amsgrad=True), loss=categorical_crossentropy, metrics=['accuracy'])
    X_single_model.compile(optimizer=Adam(lr=1e-4, decay=1e-6, amsgrad=True), loss=categorical_crossentropy, metrics=['accuracy'])
    Y_single_model.compile(optimizer=Adam(lr=1e-4, decay=1e-6, amsgrad=True), loss=categorical_crossentropy, metrics=['accuracy'])

    batch_size = 256


    early_stopping = EarlyStopping(monitor='loss', mode='min', patience=10)

    # 学習
    fit_multimodal_model = multimodal_model.fit(x=[X_train, Y_train],
                                                y=Z_train,
                                                #validation_split=0.2,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                callbacks=[early_stopping],
                                                verbose=0  # type: ignore
                                                )

    fit_X_single_model = X_single_model.fit(x=X_train, y=Z_train,
                                            #validation_split=0.2,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            callbacks=[early_stopping],
                                            verbose=0  # type: ignore
                                            ) 

    fit_Y_single_model = Y_single_model.fit(x=Y_train, y=Z_train,
                                            #validation_split=0.2,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            callbacks=[early_stopping],
                                            verbose=0  # type: ignore
                                            )

    return multimodal_model, X_single_model, Y_single_model, fit_multimodal_model, fit_X_single_model, fit_Y_single_model

# モダリティの特徴量抽出層
def X_encoder(X_dim):
    X_input = Input(shape=(X_dim, 1))

    hidden = Dense(16, activation='relu')(X_input)
    hidden = Dense(12, activation='relu')(hidden)
    hidden = Dense(10, activation='relu')(hidden)

    conv = Conv1D(10, 2, padding='same', activation='relu')(hidden)
    conv = MaxPool1D(pool_size=2, padding='same')(conv)

    X_feature = Flatten()(conv)
    X_feature = Dropout(0.5)(X_feature)

    # 単一モダリティ用の分類層
    classification = Dense(5, activation='softmax')(X_feature)
    X_single_model = Model(X_input, classification)

    return X_input, X_feature, X_single_model

def Y_encoder(Y_dim):
    Y_input = Input(shape=(Y_dim, 1))

    hidden = Dense(300, activation='relu')(Y_input)
    hidden = Dense(250, activation='relu')(hidden)
    hidden = Dense(150, activation='relu')(hidden)
    #hidden = Dense(50, activation='relu')(hidden)

    conv = Conv1D(50, 2, padding='same', activation='relu')(hidden)
    conv = MaxPool1D(pool_size=2, padding='same')(conv)

    Y_feature = Flatten()(conv)
    #Y_feature = Dropout(0.5)(Y_feature)

    # 単一モダリティ用の分類層
    classification = Dense(5, activation='softmax')(Y_feature)
    Y_single_model = Model(Y_input, classification)

    return Y_input, Y_feature, Y_single_model

# マルチモーダル分類層
def Multimodal_Classification_Layer(X_input, Y_input, X_feature, Y_feature):
    concat = Concatenate()([X_feature, Y_feature])

    classification = Dense(60, activation='relu')(concat)
    #classification = Dense(50, activation='relu')(classification)
    classification = Dense(20, activation='relu')(classification)
    classification = Dense(20, activation='relu')(classification)
    classification = Dense(20, activation='relu')(classification)

    classification = Dropout(0.5)(classification)
    output = Dense(5, activation='softmax', name='output_layer')(classification)

    multimodal_model = Model([X_input, Y_input], output)

    return multimodal_model
