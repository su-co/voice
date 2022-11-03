import os
from scipy.io import wavfile
import pyroomacoustics as pra
import numpy as np
from matplotlib import pyplot

speech_path = os.path.join("au", "speech.wav")
noise_path = os.path.join("au", "noise.wav")
speech_rate, speech = wavfile.read(speech_path)
noise_rate, noise = wavfile.read(noise_path)

'''设置房间'''
rt60 = 0.3  # 声音衰减到60dB需要0.3s
room_dim = [4, 4, 4]  # 房间大小为4*4*4m
e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)  # 返回 墙壁吸收的能量 和 允许的反射次数
room = pra.ShoeBox(room_dim, fs=speech_rate,
                   materials=pra.Material(e_absorption),
                   max_order=max_order)

'''放置信号源、噪声源、麦克风'''
room.add_source([1, 1, 1], signal=speech, delay=2)  # 让说话人推迟两秒发声，模拟噪声背景一直都在
room.add_source([2, 2, 2], signal=noise, delay=0)
mic_location = np.array([[3, 3, 3]]).transpose()
room.add_microphone(mic_location)

'''开始仿真'''
room.simulate()

'''保存结果'''
output_path = os.path.join("au", "output.wav")
room.mic_array.to_wav(output_path, norm=True, bitdepth=np.int16)

'''声音到麦克风的脉冲响应绘图'''
src_id = 0  # 声源编号
mic_id = 0  # 麦克风编号
rir = room.rir[mic_id][src_id]
pyplot.plot(np.arange(len(rir)) / room.fs, rir)
pyplot.show()
