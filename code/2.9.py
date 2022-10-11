"""
https://blog.csdn.net/xyn1996/article/details/109092502
pyAudioAnalysis教程：https://www.cnblogs.com/littlemujiang/p/pyAudioAnalysis-wen-dang.html
"""
import os
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import numpy as np
from matplotlib import pyplot as plt

input_file = os.path.join('..', 'data', 'demo3.0.wav')
rate, sig = audioBasicIO.read_audio_file(input_file)  # 读取的就是int16，跟scipy中的一样，因为它就是依赖的scipy库

'''
ShortTermFeatures.feature_extraction
返回的列是统计过的特征数据（68列，前34列是均值，后34列是标准差），行是帧数
'''
frame_len, frame_spacing = 0.025, 0.01  # 帧长与帧间隔
feature, feature_name = ShortTermFeatures.feature_extraction(sig, rate, frame_len * rate, frame_spacing * rate)
for i in range(len(feature_name)):  # 展示所有特征名称
    print(i, ':', feature_name[i])

'''以能量特征为例绘图'''
duration = len(sig) / rate  # 计算音频时长
time = np.arange(0, duration - frame_len, frame_spacing)  # 以帧构建时间轴
energy = feature[feature_name.index('energy'), :]
plt.plot(time, energy)
plt.savefig('../result/figure/energy')
plt.show()
