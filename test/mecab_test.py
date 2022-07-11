import MeCab

m = MeCab.Tagger("-d /opt/mecab/lib/mecab/dic/neologd -Owakati")        # NEologd
#m = MeCab.Tagger('-Owakati')        # ipadic

text = "ヤバイTシャツ屋さんと岡崎体育のライブに行く"
wakati = m.parse(text)

print(wakati)