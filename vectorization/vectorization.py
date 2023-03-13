# Created by RitsukiShuto on 2022/10/09.
# vectorization.py
# 音声、言語の生データをベクトル化する。
#
import pandas as pd
import glob
import re
import numpy as np

import wave
import librosa as lr
import numpy as np
import MeCab as mecab

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# 分かち書きの辞書を指定
m = mecab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/ -Owakati")       # NEologd

# wavファイルを読み込む
def load_wav(filename):
    # open wave file
    wf = wave.open(filename,'r')
    channels = wf.getnchannels()

    # load wave data
    chunk_size = wf.getnframes()
    amp  = (2**8) ** wf.getsampwidth() / 2

    binary = np.frombuffer(wf.readframes(chunk_size), 'int16')      # intに変換
    normalized_amplitude = binary / amp                             # 振幅正規化
    data = normalized_amplitude[::channels]
   
    return data

# FFTを行いパワースペクトルを計算する
def calc_pow_spectrum(wav_file, size):
    st = 1000                       # サンプリングの開始位置
    wave = load_wav(wav_file)       # wavファイルを読み込む

    return abs(np.fft.fft(np.hamming(size) * wave[st:st+size])) ** 2

# MFCCを求める
def calc_MFCC(wav_file):
    y, sr = lr.core.load(wav_file, sr=None)
    mfcc = lr.feature.mfcc(y=y, sr=sr, n_mfcc=12)

    return mfcc.mean(axis=1)

# 分かち書き
def wakatigaki(sentence):
    # 記号を削除
    code_regex = re.compile('[\t\s\n!"#$%&\'\\\\()*+,-./:;；：<=>?@[\\]^_`{|}~○｢｣「」〔〕“”〈〉笑、。'\
                            '『』【】＆＊（）＄＃＠？?！!｀＋￥¥％♪…◇→←↓↑｡･ω･｡ﾟ´∀｀ΣДｘ⑥◎©︎♡★☆▽※ゞノ〆εσ＞＜┌┘]')
    sentence = code_regex.sub('', sentence)

    # 分かち書き
    wakati = m.parse(sentence)
    wakati = wakati.rstrip('\n')

    return wakati

def calc_TF_IDF_and_to_PCA(wakati_list):
    vec_tfidf = TfidfVectorizer(max_df=0.9)
    val_tfidf = vec_tfidf.fit_transform(wakati_list)

    print("\n(発話数, 単語数)={}".format(val_tfidf.shape))

    pca = PCA(n_components=367, whiten=False)       # n_components = 次元数
    pca.fit(val_tfidf.toarray())

    print("次元削減後の次元数={}".format(pca.n_components_))

    return pca.transform(val_tfidf.toarray())

def main():
    wav_dir = "../data/wav/mixed/"             # wavファイルのディレクトリ
    save_dir = "../train_data/feature_vector/"          # 保存先

    # ベクトル化したデータを格納するための変数
    wakati_list = []
    new_meta = []

    labeled_pow_list = []
    unlabeled_pow_list = []

    labeled_MFCC_list = []
    unlabeled_MFCC_list = []

    tfidf_unlabeled = []
    tfidf_labeled = []

    # 各種データに対する処理回数のカウンタ
    cnt_prosessed_data = 0
    cnt_skip_data = 0
    cnt_UNlabeled_data = 0
    cnt_labeled_data = 0

    FILE_NUM = str(11)                          # 保存するファイル名の番号
    LEN = 0                                     # LEN文字以上のデータを学習データとして使う
    SOUND_DIM = 367                             # 音声の特徴量ベクトルはSOUND_DIM次元である
    PICKUP_EMO_LV = [1, 2, 3, 9]                # 学習に使う感情レベルを選ぶ."9"は自発対話音声

    # メタデータの読み込み
    meta_data = pd.read_csv("../data/MOY_metadata_5emo.csv", header=0)

    # テキストの分かち書きとパワースペクトルへの変換を1発話ごとに行う
    for row in meta_data.values:
        if row[3] == "{笑}" or len(row[3]) < LEN or row[4] not in PICKUP_EMO_LV:    # 声喩のみの発話とLEN文字以下の発話をスキップ
            print("[INFO]skip")     # 学習に使えない発話はスキップ
            cnt_skip_data += 1

            continue
        
        # メタデータからwavファイル名を作成
        if pd.isnull(row[1]):       # 自発対話音声と演技音声のファイルの命名規則が異なるのでここで判別。連番がないやつは演技音声
            wav_file_name = str(row[0])
        else:
            wav_file_name = str(row[0]) + "_" + str(int(row[1]))

        # パワースペクトルとMFCCを計算
        pow_spectrum = calc_pow_spectrum(wav_dir + wav_file_name + ".wav", SOUND_DIM)
        MFCC = calc_MFCC(wav_dir + wav_file_name + ".wav")

        # 分かち書きしてリストに追加
        wakati_list.append(wakatigaki(str(row[3])))

        if pd.isnull(row[5]):       # ラベルなしデータ
            print("[INFO]", wav_file_name, "is un labeled data.")     # DEBUG

            # データの先頭にファイル名を付けて、ラベルなしデータ群に追加
            unlabeled_pow_list.append(np.hstack((wav_file_name, pow_spectrum)))
            unlabeled_MFCC_list.append(np.hstack((wav_file_name, MFCC)))

            cnt_UNlabeled_data += 1

        else:                       # ラベルありデータ
            print("[INFO]", wav_file_name, "is labeled data.")     # DEBUG

            # データの先頭にファイル名を付けて、ラベルありデータ群に追加
            labeled_pow_list.append(np.hstack((wav_file_name, pow_spectrum)))
            labeled_MFCC_list.append(np.hstack((wav_file_name, MFCC)))

            cnt_labeled_data += 1

        new_meta.append(row)
        cnt_prosessed_data += 1

    pca_tfidf = calc_TF_IDF_and_to_PCA(wakati_list)     # TF-IDFを計算


    # LEN文字以下の発話を除いたメタデータを生成
    new_meta = pd.DataFrame(new_meta)
    new_meta.columns = ['fid', 'no', 'person', 'text', 'lv', 'emotion']  # type: ignore
    # FIXME: ファイル名を変更して保存
    new_meta.to_csv("../train_data/meta_data/" + FILE_NUM + "_meta.csv", index=True, header=1)  # type: ignore

    i = 0
    for row in new_meta.values:
        if pd.isnull(row[5]):
            tfidf_unlabeled.append(pca_tfidf[i][0:])

        else:
            tfidf_labeled.append(pca_tfidf[i][0:])

        i += 1

    print(f"ラベル付きデータ:{cnt_labeled_data}\nラベルなしデータ:{cnt_UNlabeled_data}")

    # テキストデータをcsvに変換して保存
    # ラベルありデータ
    labeled_tfidf = pd.DataFrame(tfidf_labeled)
    labeled_pow = pd.DataFrame(labeled_pow_list)
    labeled_mfcc = pd.DataFrame(labeled_MFCC_list)

    # ラベルなしデータ
    unlabeled_tfidf = pd.DataFrame(tfidf_unlabeled)
    unlabeled_pow = pd.DataFrame(unlabeled_pow_list)
    unlabeled_mfcc = pd.DataFrame(unlabeled_MFCC_list)

    labeled_tfidf.to_csv(save_dir + FILE_NUM + "_TF-IDF_labeled.csv", index=True, header=1)       # type: ignore
    labeled_pow.to_csv(save_dir + FILE_NUM + "_POW_labeled.csv", index=False, header=1)           # type: ignore
    labeled_mfcc.to_csv(save_dir + FILE_NUM + "_MFCC_labeled.csv", index=False, header=1)         # type: ignore
    unlabeled_tfidf.to_csv(save_dir + FILE_NUM + "_TF-IDF_un_labeled.csv", index=True, header=1)  # type: ignore
    unlabeled_pow.to_csv(save_dir + FILE_NUM + "_POW_un_labeled.csv",index=False, header=1)       # type: ignore
    unlabeled_mfcc.to_csv(save_dir + FILE_NUM + "_MFCC_un_labeled.csv", index=False, header=1)    # type: ignore

if __name__ == '__main__':
    main()