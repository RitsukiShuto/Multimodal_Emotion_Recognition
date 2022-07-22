# Created by RitsukiShuto on 2022/07/22.
# TF-IDF.py
#
from numpy import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer

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

    df = pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())

    return df


def main():
    doc_list = glob.glob("../data/wakachi/*.txt")
    for docs in doc_list:
        df = TF_IDF(docs)

        # csvを保存
        if docs == "../data/wakachi/full_wakati.txt":       # 'emotion'ラベル付き
            df.to_csv("../vector/bag-of-words/full_labeled_TF-IDF.csv", index=False)

        elif docs == "../data/wakachi/half_wakati.txt":     # 'ans_n'ラベル付き
            df.to_csv("../vector/bag-of-words/half_labeled_TF-IDF.csv", index=False)

        else:                                               # ラベルなし
            df.to_csv("../vector/bag-of-words/un_labeled_TF-IDF.csv", index=False)


if __name__ == '__main__':
    main()