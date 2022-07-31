# Created by RitsukiShuto on 2022/08/01.
# train_model.py
#
from secrets import choice
import numpy as np
import pandas as pd
import random


# メタデータのディレクトリ
meta_data = pd.read_csv("data/supervised_list.csv", header=0)

# 訓練データのディレクトリ
sound_labeled_X1 = pd.read_csv("train_data/2div/POW_labeled.csv", header=0, index_col=0)
tfidf_labeled_X2 = pd.read_csv("train_data/2div/TF-IDF_labeled_PCA.csv", header=0, index_col=0)
sound_un_labeled_X1 = pd.read_csv("train_data/2div/POW_un_labeled.csv", header=0, index_col=0)
tfidf_un_labeled_X2 = pd.read_csv("train_data/2div/TF-IDF_un_labeled_PCA.csv", header=0, index_col=0)

# データ格納
# 教師ありデータ
print("supervised sound data\n", sound_labeled_X1.head(), "\n")
print("supervised tfidf data\n", tfidf_labeled_X2.head(), "\n")

''' 教師ありデータ
supervised sound data
           0         1         2             3             4         5  ...       122       123           124           125       126       127
0  1.080402  0.847819  0.486187  2.550507e-01  1.531421e-01  0.001108  ...  0.001223  0.001108  1.531421e-01  2.550507e-01  0.486187  0.847819
1  0.041937  0.140406  0.015008  1.202276e-01  8.914298e-02  0.013332  ...  0.005039  0.013332  8.914298e-02  1.202276e-01  0.015008  0.140406
2  0.000948  0.000164  0.001978  6.681123e-04  1.655019e-04  0.000017  ...  0.000018  0.000017  1.655019e-04  6.681123e-04  0.001978  0.000164
3  0.097222  0.136776  0.038901  1.391849e-01  2.597667e-01  0.024267  ...  0.001717  0.024267  2.597667e-01  1.391849e-01  0.038901  0.136776
4  0.005315  0.001219  0.000004  7.196278e-07  4.645277e-07  0.000004  ...  0.000005  0.000004  4.645277e-07  7.196278e-07  0.000004  0.001219

[5 rows x 128 columns]

supervised tfidf data
           0         1         2         3         4         5  ...       547       548           549       550       551       552
0  0.945416  0.001287  0.001237  0.000507  0.004263  0.000474  ...  0.000026 -0.000224  3.756237e-15 -0.000624 -0.000396  0.000153
1 -0.053910 -0.015706 -0.010683 -0.006480 -0.017092 -0.009506  ... -0.002183 -0.002545  8.323981e-15 -0.001024 -0.002747 -0.004682
2 -0.052863 -0.013727 -0.010393 -0.003839 -0.011973 -0.007187  ... -0.013867 -0.003732  2.873909e-14 -0.003669  0.003164  0.009518
3  0.945416  0.001287  0.001237  0.000507  0.004263  0.000474  ...  0.000026 -0.000224  3.756237e-15 -0.000624 -0.000396  0.000153
4  0.945416  0.001287  0.001237  0.000507  0.004263  0.000474  ...  0.000026 -0.000224  3.756237e-15 -0.000624 -0.000396  0.000153

[5 rows x 553 columns]
'''

# 教師なしデータ
# データをランダムに欠損させる<試作版>
for (unX1_row, unX2_row) in zip(sound_un_labeled_X1.values, tfidf_un_labeled_X2.values):
    missing = random.choice([0, 1, 2, 3])

    if missing == 0:
        unX1_row[:] = None

    else:
        unX2_row[:] = None

print("un supervised sound data\n", sound_un_labeled_X1.head(), "\n")
print("un supervised tfidf data\n", tfidf_un_labeled_X2.head(), "\n")
print("missing data:", sound_un_labeled_X1.isnull().sum().sum() / 128)

''' 教師なしデータ
un supervised sound data
           0         1         2         3         4         5  ...       122       123       124       125       126       127
0       NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN       NaN       NaN       NaN       NaN
1  1.364893  0.570444  0.025784  0.000383  0.000145  0.001261  ...  0.000722  0.001261  0.000145  0.000383  0.025784  0.570444
2  0.000429  0.000445  0.005929  0.004049  0.000537  0.000155  ...  0.000288  0.000155  0.000537  0.004049  0.005929  0.000445
3       NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN       NaN       NaN       NaN       NaN
4  0.000671  0.004843  0.004340  0.000632  0.000012  0.000007  ...  0.000004  0.000007  0.000012  0.000632  0.004340  0.004843

[5 rows x 128 columns]

un supervised tfidf data
           0         1         2         3         4         5  ...       547       548           549       550       551       552
0 -0.052234 -0.012430 -0.010070 -0.005629 -0.012027 -0.005479  ... -0.000151 -0.001849  1.288235e-14 -0.001660 -0.000535  0.004911
1       NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN           NaN       NaN       NaN       NaN
2       NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN           NaN       NaN       NaN       NaN
3 -0.054916 -0.016193 -0.012369 -0.008113 -0.013379 -0.005650  ...  0.002097  0.000809  6.895436e-15 -0.001321  0.001770 -0.004761
4       NaN       NaN       NaN       NaN       NaN       NaN  ...       NaN       NaN           NaN       NaN       NaN       NaN

[5 rows x 553 columns]

missing data: 1327.0
'''

