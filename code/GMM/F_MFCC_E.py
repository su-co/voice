import librosa
import numpy as np

'''提取MFCC特征
详情参考 http://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
librosa.core.load 读取wav文件（返回浮点数的做了32767的归一化），在之前我使用了scipy读取（返回整数）
y=表示数据
sr=None表示保持原采样率
hop_length可以简单地理解为帧间隔（1s/100=10ms）
win_length理解为帧长（1s/40=25ms）
n_fft快速傅里叶变换维数
只关注fmin到fmax的频率范围
np.shape(features) = (40*帧数)，也就是说，每一帧都有一个MFCC特征！！！！！
'''


def f_e(file_path):  # 根据路径提取wav文件MFCC特征
    normalized_signal, rate = librosa.core.load(file_path, sr=None)  # 读取信号和采样率
    features = librosa.feature.mfcc(y=normalized_signal, sr=rate, n_mfcc=40, hop_length=rate // 100,  # 提取特征
                                    win_length=rate // 40, n_fft=2048, fmin=100, fmax=4000)  # 帧长25ms，帧间隔10ms
    # print(np.shape(features))
    features_cut = features[:, 0:500]  # 为什么需要切片（因为，GMM在评分的时候，需要权限者和测试者的特征大小相同，但是音频长度
    return features_cut  # 不同，无法得到相同大小的特征）为什么不使用reshape（reshape不能够改变元素个数）
