import pytube
import os
import librosa
from matplotlib import pyplot as plt

# '''下载mp4文件'''
# url = "https://www.youtube.com/watch?v=FM7MFYoylVs&ab_channel=ChainsmokersVEVO"
# output_path = os.path.join("data")
# file_name = "music.mp4"
# pytube.YouTube(url).streams.filter(
#     only_audio=True, file_extension="mp4")[0].download(output_path=output_path, filename=file_name)
#
# '''转换为相应格式的wav文件'''
# os.system("ffmpeg -i ./data/music.mp4 ./data/music.wav")  # 格式转换（跑代码的时候报错，说找不到文件，少了.表示当前目录）
# os.system("sox ./data/music.wav -r 16000 -b 16 -e signed-integer ./data/music_change.wav remix 1")  # 修改为单声道、16000Hz

'''提取MFCC特征
详情参考 http://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
librosa.core.load 读取wav文件（返回浮点数的做了32767的归一化），在之前我使用了scipy读取（返回整数）
y=表示数据
sr=None表示保持原采样率
hop_length可以简单地理解为帧间隔（1s/100=10ms）
win_length理解为帧长
n_fft快速傅里叶变换维数
只关注fmin到fmax的频率范围
'''
normalized_signal, rate = librosa.core.load("./data/music_change.wav", sr=None)
features = librosa.feature.mfcc(y=normalized_signal, sr=rate, n_mfcc=40, hop_length=rate // 100,
                                win_length=rate // 40, n_fft=512, fmin=100, fmax=4000)

'''绘制特征图'''
plt.matshow(features)
plt.axis('tight')
plt.savefig("./result/mfcc")
plt.show()
