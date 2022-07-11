# Created by RitsukiShuto on 2022/07/10.
# wakati
#
import pandas as pd
import re

import MeCab as mecab
from gensim.corpora import Dictionary
from gensim import matutils as mtu

wakati_list = []
#m = mecab.Tagger("-d /opt/mecab/lib/mecab/dic/neologd -Owakati")       # NEologd
m = mecab.Tagger('-Owakati')        # ipadic

# csvを読み込む
df = pd.read_csv('../train_data/trans/text-only.csv', encoding='utf-8', header=0)

# 分かち書きを行う発話を選定し、必要な処理を行う
for row in df.values:
    sentence = str(row)

    if sentence != "['{笑}']":      # 「{笑}」の発話はスキップ
        # 記号を削除
        code_regex = re.compile('[\t\s\n!"#$%&\'\\\\()*+,-./:;；：<=>?@[\\]^_`{|}~○｢｣「」〔〕“”〈〉笑'\
                                '『』【】＆＊（）＄＃＠？！｀＋￥¥％♪…◇→←↓↑｡･ω･｡ﾟ´∀｀ΣДｘ⑥◎©︎♡★☆▽※ゞノ〆εσ＞＜┌┘]')
        sentence = code_regex.sub('', sentence)

        # 分かち書き
        wakati = m.parse(sentence)
        wakati = wakati.rstrip('\n')

        wakati_list.append(wakati)

df = pd.DataFrame(wakati_list)
df.to_csv("../ipadic_wakati.txt", index=False)
print(df)       # DEBUG