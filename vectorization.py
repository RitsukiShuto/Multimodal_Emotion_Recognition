# Created by RitsukiShuto on 2022/10/09.
# vectorization.py
# 音声、言語の生データをベクトル化する。
# 
import pandas as pd
import glob
import re
import numpy as np

import wave
import MeCab as mecab

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# 分かち書きの辞書を指定
m = mecab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/ -Owakati")       # NEologd
#m = mecab.Tagger('-Owakati')        # ipadic

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
    st = 1000                       # サンプリングする開始位置
    fs = 44100                      # サンプリングレート

    wave = load_wav(wav_file)
    # 切り出した波形データ(np.hamming(size) * wave[st:st+size])にFFTを行う
    return abs(np.fft.fft(np.hamming(size) * wave[st:st+size])) ** 2

# 分かち書き
def wakatigaki(sentence):
    # 記号を削除
    code_regex = re.compile('[\t\s\n!"#$%&\'\\\\()*+,-./:;；：<=>?@[\\]^_`{|}~○｢｣「」〔〕“”〈〉笑、。'\
                            '『』【】＆＊（）＄＃＠？！｀＋￥¥％♪…◇→←↓↑｡･ω･｡ﾟ´∀｀ΣДｘ⑥◎©︎♡★☆▽※ゞノ〆εσ＞＜┌┘]')
    sentence = code_regex.sub('', sentence)

    # 分かち書き
    wakati = m.parse(sentence)
    wakati = wakati.rstrip('\n')

    return wakati

def calc_TF_IDF_and_to_PCA(wakati_list):
    vec_tfidf = TfidfVectorizer(max_df=0.9)
    val_tfidf = vec_tfidf.fit_transform(wakati_list)

    print("\n(発話数, 単語数)={}".format(val_tfidf.shape))

    pca = PCA(n_components=0.7, whiten=False)
    pca.fit(val_tfidf.toarray())

    print("次元削減後の次元数={}".format(pca.n_components_))

    return pca.transform(val_tfidf.toarray())

def main():
    wav_dir = "data/wav/OGVC_vol1/all_data/"    # wavファイルのディレクトリ
    save_dir = "train_data/OGVC_vol1/"          # 保存先

    # ベクトル化したデータを格納するための変数
    wakati_list = []
    labeled_pow_list = []
    unlabeled_pow_list = []
    new_meta = []

    # 各種データに対する処理回数のカウンタ
    cnt_prosessed_data = 0
    cnt_skip_data = 0
    cnt_UNlabeled_data = 0
    cnt_labeled_data = 0

    LEN = 0             # LEN文字以上のデータを学習データとして使う
    SOUND_DIM = 64      # 音声の特徴量ベクトルはSOUND_DIM次元である

    # メタデータの読み込み
    # INFO: 読み込ませるデータセットはここで変更する。
    meta_data = pd.read_csv("data/OGVC_Vol1_supervised_4emo.csv", header=0)
    #meta_data = pd.read_csv("/data/OGVC2_metadata.csv", header=0)

    # 音声データをパワースペクトルに変換
    # 言語データを分かち書き
    for row in meta_data.values:
        if row[5] == "{笑}" and len(row[5]) < LEN:           # 声喩のみの発話とLEN文字以下の発話をスキップ
            print("[INFO]skip")
            cnt_skip_data += 1

            break

        wav_file_name = str(row[0]) + "_" + str(row[1])     # FFTを行うwavファイルを指定

        # パワースペクトルを計算
        pow_spectrum = calc_pow_spectrum(wav_dir + wav_file_name + ".wav", SOUND_DIM)

        # 分かち書きしてリストに追加
        wakati_list.append(wakatigaki(str(row[5])))

        if pd.isnull(row[9]):       # ラベルなしデータ
            print("[INFO]", wav_file_name, "is un labeled data.")     # DEBUG

            # データの先頭にファイル名を付けて、ラベルなしデータ群に追加
            unlabeled_pow_list.append(np.hstack((wav_file_name, pow_spectrum)))

            cnt_UNlabeled_data += 1

        else:                       # ラベルありデータ
            print("[INFO]", wav_file_name, "is labeled data.")     # DEBUG

            # データの先頭にファイル名を付けて、ラベルありデータ群に追加
            labeled_pow_list.append(np.hstack((wav_file_name, pow_spectrum)))

            cnt_labeled_data += 1

        new_meta.append(row)
        cnt_prosessed_data += 1

    # LEN文字以下の発話を除いたメタデータを生成
    new_meta = pd.DataFrame(new_meta)
    new_meta.columns = ["fid", "no", "start", "end", "person", "text", "ans1", "ans2", "ans3", "emotion"]  # type: ignore
    new_meta.to_csv("train_data/meta_data/LEN7_meta.csv", index=True, header=1)  # type: ignore

    pca_tfidf = calc_TF_IDF_and_to_PCA(wakati_list)     # TF-IDFを計算

    tfidf_unlabeled = []
    tfidf_labeled = []

    i = 0
    for row in new_meta.values:
        if pd.isnull(row[9]):
            tfidf_unlabeled.append(pca_tfidf[i][0:])

        else:
            tfidf_labeled.append(pca_tfidf[i][0:])

        i += 1

    # テキストデータをcsvに変換して保存
    # ラベルありデータ
    df1 = pd.DataFrame(tfidf_labeled)
    df2 = pd.DataFrame(labeled_pow_list)
    df1.to_csv(save_dir+"TF-IDF_labeled.csv", index=True, header=1)  # type: ignore
    df2.to_csv(save_dir+"POW_labeled.csv", index=False, header=1)  # type: ignore

    # ラベルなしデータ
    df3 = pd.DataFrame(tfidf_unlabeled)
    df4 = pd.DataFrame(unlabeled_pow_list)
    df3.to_csv(save_dir+"TF-IDF_un_labeled.csv", index=True, header=1)  # type: ignore
    df4.to_csv(save_dir+"POW_un_labeled.csv", index=False, header=1)  # type: ignore

if __name__ == '__main__':
    main()