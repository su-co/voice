import os
import scipy
import wave
import numpy as np
import math
from sgn import sgn
from matplotlib import pyplot as plt
from shutil import copyfile
import sys

'''读取wav文件信息'''
input_file = os.path.join('data', 'demo3.0.wav')
rate, sig = scipy.io.wavfile.read(input_file)  # 读取采样值
# print(max(abs(sig)))  # 也可以采用最大绝对值进行归一化处理
sig = sig / 32767  # 归一化处理（一般都是16位PCM，所以除32767）
f = wave.open(input_file)
bit_type = f.getsampwidth() * 8  # 读取编码位数
mu = 2 ** bit_type - 1  # 根据编码位数确定mu值
denominator = math.log(1 + mu)  # 根据mu值确定mu-law变换中的分母
f.close()

'''
mu-law变换，直观化展示
'''
sig_change = np.zeros(np.shape(sig))  # 创建同样形状的np数组，保存修改过后的sig
for i in range(len(sig)):
    numerator = math.log(1 + mu * abs(sig[i]))  # 分子
    sig_change[i] = sgn(sig[i]) * (numerator / denominator)
plt.plot(sig, sig)  # 最初的线性展示
plt.plot(sig, sig_change)  # 转换后的非线性展示
plt.savefig("./result/Nonlinear_coding")
plt.show()

'''
保存改变后的文件
'''
output_file = os.path.join('data', 'demo3.0mu-law.wav')
sig_change = sig_change * 255  # 因为scipy.io.wavfile.write就是根据数据类型决定编码方式
sig_uint8 = sig_change.astype('uint8')  # 更改类型
try:  # 复制文件
    copyfile(input_file, output_file)
except IOError as e:
    print("Unable to copy file. %s" % e)
    exit(1)
except:
    print("Unexpected error:", sys.exc_info())
    exit(1)
print("File copy done!")
scipy.io.wavfile.write(output_file, rate, sig_uint8)  # 写入更改过后的信号
