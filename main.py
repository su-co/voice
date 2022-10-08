import os
import scipy
import librosa
import numpy as np
from matplotlib import pyplot

'''读取wav文件相关信息'''
data_file = os.path.join('data', 'demo3.0.wav')
rate, data = scipy.io.wavfile.read(data_file)  # 读取波形图
duration = librosa.get_duration(filename=data_file)  # 获取wav文件播放时长
# print(rate, data)
# print(duration)

'''进行短时傅里叶变换（分帧、加窗、快速傅里叶变换）
参数介绍：
x为采样值numpy数组
fs为采样率
window为加窗函数
nperseg为每帧的个数
noverlap为相邻帧重叠个数
return_onesided是否返回一边
boundary=zeros选择用0填充填充两端（当采样值数目不能够整除nperseg时）为了保证帧的完整性
padded=True用0进行填充，保证帧的完整性（发生在boundary之后）'''
frames, frame_spacing = 25.0, 10.0  # 设置帧长和帧间隔，分别设置为25ms和10ms
f, t, tfs = scipy.signal.stft(  # 天坑！！！！不同数目的信道，返回的tfs形状不同，会导致报错！！！
    x=data, fs=rate, window='hann', nperseg=int((frames / 1000.0) * rate),
    noverlap=int(((frame_spacing - frames) / 1000.0) * rate),
    return_onesided=True, boundary='zeros', padded=True, axis=-1
)

'''绘制图谱'''
number_of_sampling_points = len(data)
# print(type(data)) #因为一开始不知道time应该设置什么数据类型，所以测试了一下
time = np.arange(0.0, duration, 1.0 / rate)  # 列表不能是浮点数，所以必须使用np
# print(len(data), len(time))
pyplot.plot(time, data)  # 最开始使用的3.6版本报错，版本太高
pyplot.show()  # 绘制波形图
# print(len(f), len(t), np.shape(tfs))
pyplot.pcolormesh(t, f, np.abs(tfs), vmin=0, vmax=1, shading='gouraud')
pyplot.show()  # 绘制时频谱
