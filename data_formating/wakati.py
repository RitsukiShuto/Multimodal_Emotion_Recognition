# Created by RitsukiShuto on 2022/07/10.
# wakati
#
import pandas as pd
import re

import MeCab as mecab
from gensim.corpora import Dictionary
from gensim import matutils as mtu

full_labeled_wakati = []
half_labeled_wakati = []
un_labeled_wakati = []

m = mecab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/ -Owakati")       # NEologd
#m = mecab.Tagger('-Owakati')        # ipadic

def wakatigaki(sentence):
    # 記号を削除
    code_regex = re.compile('[\t\s\n!"#$%&\'\\\\()*+,-./:;；：<=>?@[\\]^_`{|}~○｢｣「」〔〕“”〈〉笑'\
                            '『』【】＆＊（）＄＃＠？！｀＋￥¥％♪…◇→←↓↑｡･ω･｡ﾟ´∀｀ΣДｘ⑥◎©︎♡★☆▽※ゞノ〆εσ＞＜┌┘]')
    sentence = code_regex.sub('', sentence)

    # 分かち書き
    wakati = m.parse(sentence)
    wakati = wakati.rstrip('\n')

    return wakati

def main():
    cnt_un_labeled = 0        # init var
    cnt_half_labeled = 0
    cnt_full_labeled = 0

    # csvを読み込む
    df = pd.read_csv('../data/supervised_list.csv', encoding='utf-8', header=0)

    # 分かち書きを行う発話を選定し、必要な処理を行う
    for row in df.values:
        sentence = str(row[5])
        print(sentence)

        if pd.isnull(row[6]):      # ラベルなし
            if row[5] == "{*}":    # 語素はスキップ
                print("[skip]", row[5])

            else:
                print("[run wakatigaki()] UN LABELED\n")
                wakati = wakatigaki(sentence)
                un_labeled_wakati.append(wakati)
                cnt_un_labeled += 1

        elif pd.isnull(row[9]):    # 'ans_n'のみラベルあり
            print("[run wakatigaki()] HALF LABELED\n")
            wakati = wakatigaki(sentence)
            half_labeled_wakati.append(wakati)
            cnt_half_labeled += 1

        else:                      # 'emotion'ラベルあり
            print("[run wakatigaki()] FULL LABELED\n")
            wakati = wakatigaki(sentence)
            full_labeled_wakati.append(wakati)
            cnt_full_labeled += 1

    # CSVを保存
    df_un_wakati = pd.DataFrame(un_labeled_wakati)
    df_un_wakati.to_csv("../data/wakachigaki/un_labeled_wakati.txt", index=False)

    df_half_wakati = pd.DataFrame(half_labeled_wakati)
    df_half_wakati.to_csv("../data/wakachigaki/half_wakati.txt", index=False)

    df_full_wakati = pd.DataFrame(full_labeled_wakati)
    df_full_wakati.to_csv("../data/wakachigaki/full_wakati.txt", index=False)

    # DEBUG
    print("data=", cnt_un_labeled + cnt_half_labeled + cnt_full_labeled)
    print("un labeled=", cnt_un_labeled)
    print("half labeled=", cnt_half_labeled)
    print("full labeled=", cnt_full_labeled)

if __name__ == '__main__':
    main()