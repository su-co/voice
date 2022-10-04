import os
import scipy

data_file = os.path.join('..', 'data', 'demo.wav')
rate, data = scipy.io.wavfile.read(data_file)  # 读取wav文件
print(rate)
print(data)
