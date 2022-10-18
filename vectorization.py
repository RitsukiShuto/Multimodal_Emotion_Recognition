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
    data = wf.readframes(chunk_size)        # バイナリ読み込み
    data = np.frombuffer(data,'int16')      # intに変換
    data = data / amp                       # 振幅正規化
    data = data[::channels]
   
    return data

# FFTを行いパワースペクトルを計算する
def pow_spectrum(wav_file, size):
    st = 1000                       # サンプリングする開始位置
    fs = 44100                      # サンプリングレート
    d = 1.0 / fs                    # サンプリングレートの逆数

    hammingWindow = np.hamming(size)                    # ハミング窓
    freqList = np.fft.fftfreq(size, d)
    wave = load_wav(wav_file)
    windowedData = hammingWindow * wave[st:st+size]     # 切り出した波形データ（窓関数あり）
    data = np.fft.fft(windowedData)
    data = abs(data) ** 2

    return data

# 分かち書き
def wakatigaki(sentence):
    # 記号を削除
    code_regex = re.compile('[\t\s\n!"#$%&\'\\\\()*+,-./:;；：<=>?@[\\]^_`{|}~○｢｣「」〔〕“”〈〉笑、。'\
                            '『』【】＆＊（）＄＃＠？！｀＋￥¥％♪…◇→←↓↑｡･ω･｡ﾟ´∀｀ΣДｘ⑥◎©︎♡★☆▽※ゞノ〆εσ＞＜┌┘]')
    sentence = code_regex.sub('', sentence)

    # 分かち書き
    wakati = m.parse(sentence)
    wakati = wakati.rstrip('\n')

    #print("[Successfully executed wakatigaki()]", "\t", wakati, "\n")

    return wakati

def TF_IDF(wakati_list):
    vec_tfidf = TfidfVectorizer(max_df=0.9)
    X = vec_tfidf.fit_transform(wakati_list)

    # DEBUG
    #print('Vocabulary size: {}'.format(len(vec_tfidf.vocabulary_)))
    #print('Vocabulary content: {}'.format(vec_tfidf.vocabulary_))

    print("\n(発話数, 単語数)={}".format(X.shape))

    return X

def _PCA(X):
    pca = PCA(n_components=0.7, whiten=False)
    pca.fit(X.toarray())

    print("次元削減後の次元数={}".format(pca.n_components_))

    x = pca.transform(X.toarray())

    return x

def main():
    # メタデータの読み込み
    # INFO: 読み込ませるデータセットはここで変更する。
    meta_data = pd.read_csv("data/OGVC_Vol1_supervised.csv", header=0)
    #meta_data = pd.read_csv("/data/OGVC2_metadata.csv", header=0)

    # 学習に用いるデータの文字数の基準値
    LEN = 0

    # wavファイルのディレクトリ
    wav_dir = "data/wav/OGVC_vol1/all_data/"

    wakati_list = []
    labeled_pow_list = []
    unlabeled_pow_list = []
    new_meta = []

    cnt_prosessed_data = 0
    cnt_skip_data = 0
    cnt_UNlabeled_data = 0
    cnt_labeled_data = 0


    # 音声データをパワースペクトルに変換
    # 言語データを分かち書き
    for row in meta_data.values:
        if row[5] != "{笑}" and len(row[5]) > LEN:           # 声喩のみの発話とLEN文字以下の発話をスキップ
            new_meta.append(row)
            cnt_prosessed_data += 1

            if pd.isnull(row[9]):       # ラベルなしデータ
                cnt_UNlabeled_data += 1

                wav_file_name = str(row[0]) + "_" + str(row[1])

                print("[INFO]", wav_file_name, "is un labeled data.")     # DEBUG

                # パワースペクトルを取得
                pow_spect = pow_spectrum(wav_dir + wav_file_name + ".wav", 64)
                pow_spect = np.hstack((wav_file_name, pow_spect))
                unlabeled_pow_list.append(pow_spect)

                # 分かち書き
                sentence = str(row[5])
                doc = wakatigaki(sentence)
                wakati_list.append(doc)                 # テキストのみ抽出

            else:                       # ラベルありデータ
                cnt_labeled_data += 1

                wav_file_name = str(row[0]) + "_" + str(row[1])

                print("[INFO]", wav_file_name, "is labeled data.")     # DEBUG

                # パワースペクトルを取得
                pow_spect = pow_spectrum(wav_dir + wav_file_name + ".wav", 64)
                pow_spect = np.hstack((wav_file_name, pow_spect))
                labeled_pow_list.append(pow_spect)
                
                # 分かち書き
                sentence = str(row[5])
                doc = wakatigaki(sentence)
                wakati_list.append(doc)                 # テキストのみ抽出

        else:
            print("[INFO]skip")
            cnt_skip_data += 1

    # LEN文字以下の発話を除いたメタデータを生成
    new_meta = pd.DataFrame(new_meta)
    new_meta.columns = ["fid", "no", "start", "end", "person", "text", "ans1", "ans2", "ans3", "emotion"]
    new_meta.to_csv("train_data/meta_data/LEN7_meta.csv", index=True, header=1)

    tfidf = TF_IDF(wakati_list)     # TF-IDFを計算
    pca_tfidf = _PCA(tfidf)         # PCA

    tfidf_unlabeled = []
    tfidf_labeled = []

    i = 0
    for row in new_meta.values:
        if pd.isnull(row[9]):
            tfidf_unlabeled.append(pca_tfidf[i][0:])

        else:
            tfidf_labeled.append(pca_tfidf[i][0:])

        i += 1

    save_dir = "train_data/OGVC_vol1/"

    df1 = pd.DataFrame(unlabeled_pow_list)
    df2 = pd.DataFrame(labeled_pow_list)
    df3 = pd.DataFrame(tfidf_labeled)
    df4 = pd.DataFrame(tfidf_unlabeled)

    df1.to_csv(save_dir+"POW_un_labeled.csv", index=False, header=1)
    df2.to_csv(save_dir+"POW_labeled.csv", index=False, header=1)
    df3.to_csv(save_dir+"TF-IDF_labeled.csv", index=True, header=1)
    df4.to_csv(save_dir+"TF-IDF_un_labeled.csv", index=True, header=1)

if __name__ == '__main__':
    main()