from gensim.corpora import Dictionary
from gensim import matutils as mtu

morphemes = ['私 は ラーメン が 好き です 。',
             '私 は 餃子 が 好き です 。',
             '私 は ラーメン が 嫌い です 。']

# 辞書を作る
dct = Dictionary()
for line in morphemes:
    print(line)
    # 辞書の更新
    # All tokens should be already tokenized and normalized.
    dct.add_documents([line.split()])

word2id = dct.token2id # 単語 -> ID
print(word2id)
bow_set = []

# 文をBoWに変換
for line in morphemes:
    # [(word ID, word frequency)]
    bow_format = dct.doc2bow(line.split())
    bow_set.append(bow_format)
    print(line)
    print("BoW format: (word ID, word frequency)")
    print(bow_format)
    bow = mtu.corpus2dense([bow_format], num_terms=len(dct)).T[0]
    print("BoW")
    print(bow)
    # numpyからlistに変える
    print(bow.tolist())
    # intにする
    print(list(map(int, bow.tolist())))