import MeCab

m = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/ -Owakati")        # NEologd
#m = MeCab.Tagger('-Owakati')        # ipadic

text = "ヤバイTシャツ屋さんと岡崎体育のライブに行く"
wakati = m.parse(text)

print(wakati)