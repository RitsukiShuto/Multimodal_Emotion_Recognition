# Created by RitsukiShuto on 2022/06/28.
# 発話ごとにwavファイルを分割する。その際、{笑}のみの音声は除外する。
#
from pydub import AudioSegment
from pydub.silence import split_on_silence

import csv
import os