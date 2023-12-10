import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# 1秒で0.001秒毎のSin波とwhiteノイズの合成波を作成
t = np.arange(0, 1, 0.001)
sin_wave = np.sin(2 * np.pi * 5 * t)  # 5HzのSin波を例として作成
white_noise = np.random.normal(0, 1, len(t))  # 平均0、標準偏差1

# 合成波
composite_wave = sin_wave + white_noise

# 窓関数
windows_names = ['boxcar', 'hamming', 'hann', 'gaussian']
windows_functions = [windows.boxcar(len(t)), \
                     windows.hamming(len(t)), \
                     windows.hann(len(t)), \
                     windows.gaussian(len(t), std=150)]

# グラフ表示
fig, axs = plt.subplots(4, 1, figsize=(7, 7))
for i in range(4):
    axs[i].plot(t, composite_wave * windows_functions[i])
    axs[i].set_title(windows_names[i])
plt.tight_layout()
plt.show()