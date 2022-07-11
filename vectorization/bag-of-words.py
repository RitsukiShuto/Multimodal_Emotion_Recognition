# Created by RitsukiShuto on 2022/07/10.
# bag-of-words
#
import csv
import re

import MeCab as mecab
from gensim.corpora import Dictionary
from gensim import matutils as mtu

words_all = []
words = []
cnt_speech = 0

# csvを読み込む
words_all = open("../wakati.csv", "r", encoding="utf-8", errors="", newline="")       # TODO: 変更せよ
f = csv.reader(words_all, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
header = next(f)    # ヘッダをスキップ
print(words_all)

# 辞書を作る
dct = Dictionary()

for sentence in words_all:
    line = str(sentence)

    # 辞書の更新
    # All tokens should be already tokenized and normalized.
    dct.add_documents([line.split()])

word2id = dct.token2id # 単語 -> ID
print(word2id)

bow_set = []
# 文をBoWに変換
for sentence in words_all:
    line = str(sentence)

    # [(word ID, word frequency)]
    bow_format = dct.doc2bow(line.split())
    bow_set.append(bow_format)

    print(line)
    print("BoW format: (word ID, word frequency)")
    print(bow_format)

    bow = mtu.corpus2dense([bow_format], num_terms=len(dct)).T[0]

    print("BoW")
    print(bow)

    bow.tolist()        # numpyからlistに変える
    print(list(map(int, bow.tolist())))