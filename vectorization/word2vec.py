# Created by RitsukiShuto on 2022/06/28.
# word2vec
#
from gensim.models import word2vec

# word2vec
doc = word2vec.LineSentence("../wakati.txt")

model = word2vec.Word2Vec(doc, size=256, min_count=5, window=5, iter=3)    # TODO: パラメータ調整
model.save("../word2vec.gensim.model")

# 簡易テスト
word_vec = model.wv["ログイン"]
print(word_vec)

similar_words = model.wv.most_similar(positive=["ログイン"], topn=9)
print(*[" ".join([v, str("{:.2f}".format(s))]) for v, s in similar_words], sep="\n")