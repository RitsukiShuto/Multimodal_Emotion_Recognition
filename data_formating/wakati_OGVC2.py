# Created by RitsukiShuto on 2022/07/10.
# wakati
#
import pandas as pd
import re

import MeCab as mecab
from gensim.corpora import Dictionary
from gensim import matutils as mtu

# 辞書を指定
m = mecab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/ -Owakati")       # NEologd
#m = mecab.Tagger('-Owakati')        # ipadic

# 分かち書き関数
# main()から"sentence[発話文章]"が渡される。
def wakatigaki(sentence):
    # 記号を削除
    code_regex = re.compile('[\t\s\n!"#$%&\'\\\\()*+,-./:;；：<=>?@[\\]^_`{|}~○｢｣「」〔〕“”〈〉笑'\
                            '『』【】＆＊（）＄＃＠？！｀＋￥¥％♪…◇→←↓↑｡･ω･｡ﾟ´∀｀ΣДｘ⑥◎©︎♡★☆▽※ゞノ〆εσ＞＜┌┘]')
    sentence = code_regex.sub('', sentence)

    # 分かち書き
    wakati = m.parse(sentence)
    wakati = wakati.rstrip('\n')

    print("[Successfully executed wakatigaki()]", "\t", wakati, "\n")

    return wakati


def main():

    wakati_list = []         # 分かち書き後の文章を格納する変数

    # csvを読み込む
    df = pd.read_csv('../data/OGVC2_metadata.csv', encoding='utf-8', header=0)

    # 分かち書きを行う発話を選定し、必要な処理を行う
    for row in df.values:
        sentence = str(row[4])                  # 'emotion'ラベルあり
        print("[run wakatigaki()] LABELED", "\t", sentence)
        wakati = wakatigaki(sentence)
        wakati_list.append(wakati)

    # CSVを保存
    df_labeled_wakati = pd.DataFrame(wakati_list)
    df_labeled_wakati.to_csv("../data/wakachi/OGVC_vol2/wakati_OGVC2.txt", index=False)

if __name__ == '__main__':
    main()