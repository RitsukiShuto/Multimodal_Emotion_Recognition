# Created by RitsukiShuto on 2022/07/22.
# TF-IDF.py
#
from numpy import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import glob
import csv
import pandas as pd
import numpy as np

def TF_IDF(docs):
    # csvを読み込む
    text = open(docs, "r", encoding="utf-8", errors="", newline="")       # TODO: 変更せよ
    f = csv.reader(text, delimiter=",", doublequote=True,
                         lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    header = next(f)    # ヘッダをスキップ

    vec_tfidf = TfidfVectorizer(max_df=0.9)
    X = vec_tfidf.fit_transform(text)

    # DEBUG
    #print('Vocabulary size: {}'.format(len(vec_tfidf.vocabulary_)))
    #print('Vocabulary content: {}'.format(vec_tfidf.vocabulary_))

    print("(発話数, 単語数)={}".format(X.shape))

    return X


def _PCA(X):
    pca = PCA(n_components=0.7, whiten=False)
    pca.fit(X.toarray())

    print("次元削減後の次元数={}".format(pca.n_components_))

    x = pca.transform(X.toarray())

    return x


def split_data(x):
    TF_IDF_labeled = []
    TF_IDF_un_labeled = []

    cnt_labeled_data = 0
    cnt_un_labeled_data = 0

    meta_data = pd.read_csv("../data/OGVC_Vol2_supervised.csv", header=0)

    i = 0
    for row in meta_data.values:
        #print(row[9], speech[0:4], i, "/8365")
        if pd.isnull(row[2]):
            print("[UN LABELED]{}/8365\n{}\n".format(i+1, x[i][0:6]))
            TF_IDF_un_labeled.append(x[i][0:])
            cnt_un_labeled_data += 1

        else:
            print("[LABELED]{}/8365\n{}\n".format(i+1, x[i][0:6]))
            TF_IDF_labeled.append(x[i][0:])
            cnt_labeled_data += 1

        i += 1

    df1 = pd.DataFrame(TF_IDF_labeled)
    
    df1.to_csv("../train_data/OGVC_vol2/TF-IDF_labeled_PCA.csv", index=True, header=1)
    

    print("data = {}".format(cnt_un_labeled_data + cnt_labeled_data))
    print("labeled_data = {}".format(cnt_labeled_data))
    print("un labeled data = {}".format(cnt_un_labeled_data))


def main():
    docs = "../data/wakachi/OGVC_vol2/wakati_OGVC2.txt"

    X = TF_IDF(docs)
    x = _PCA(X)

    df1 = pd.DataFrame(x)
    split_data(x)
    

if __name__ == '__main__':
    main()