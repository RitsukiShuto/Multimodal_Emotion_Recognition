# Created by RitsukiShuto on 2022/07/22.
# TF-IDF.py
#
from numpy import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import glob
import csv
import pandas as pd

def TF_IDF(docs):
    # csvを読み込む
    text = open(docs, "r", encoding="utf-8", errors="", newline="")       # TODO: 変更せよ
    f = csv.reader(text, delimiter=",", doublequote=True,
                         lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    header = next(f)    # ヘッダをスキップ

    vec_tfidf = TfidfVectorizer(max_df=0.9)
    X = vec_tfidf.fit_transform(text)

    # DEBUG
    print('Vocabulary size: {}'.format(len(vec_tfidf.vocabulary_)))
    #print('Vocabulary content: {}'.format(vec_tfidf.vocabulary_))

    print(X.shape)

    return X


def _PCA(X):
    pca = PCA(n_components=0.7, whiten=False)
    pca.fit(X.toarray())

    print(pca.n_components_)

    x = pca.transform(X.toarray())

    return x


def main():
    doc_list = glob.glob("../data/wakachi/2div/*.txt")

    for docs in doc_list:
        X = TF_IDF(docs)
        x = _PCA(X)

        df1 = pd.DataFrame(X.toarray())
        #df1 = df1.columns=list(range(len(df1.columns)))

        df2 = pd.DataFrame(x)
        #df2 = df2.columns=list(range(len(df2.columns)))


        # csvを保存
        if docs == "../data/wakachi/2div/labeled_wakati.txt":       # 'emotion'ラベル付き
            df1.to_csv("../vector/bag-of-words/labeled_TF-IDF.csv", index=False, header=0)
            df2.to_csv("../train_data/2div/TF-IDF_labeled_PCA.csv", index=False, header=0)

        else:                                               # ラベルなし
            df1.to_csv("../vector/bag-of-words/un_labeled_TF-IDF.csv", index=False, header=0)
            df2.to_csv("../train_data/2div/TF-IDF_un_labeled_PCA.csv", index=False, header=0)                                          # ラベルなし
            

if __name__ == '__main__':
    main()