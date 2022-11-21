# Crreated by RitsukiShuto on 2022/07/18.
# MFCC.py
#
import pandas as pd
import glob
import librosa as lr

wav_dir = glob.glob("../data/wav/labeled/*.wav")
save_dir = "../vector/MFCC/"
ceps_list = []

# MFCC
for wav_list in wav_dir:
    print(wav_list)

    y, sr = lr.core.load(wav_list, sr=None)
    mfcc = lr.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    ceps = mfcc.mean(axis=1)

    ceps_list.append(ceps)

# データフレームを作成
df_ceps = pd.DataFrame(ceps_list)
columns_name = []
for i in range(20):
    columns_name_tmp = 'dct{0}'.format(i)
    columns_name.append(columns_name_tmp)

df_ceps.columns = columns_name
df_ceps.to_csv(save_dir + "MFCC_labeled.csv", index=False)