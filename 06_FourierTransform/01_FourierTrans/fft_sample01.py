import numpy as np
import matplotlib.pyplot as plt

# サンプリング周波数
fs = 1000

# 時間軸
t = np.arange(0, 1, 1/fs)

# 周波数と振幅
f1 = 10
f2 = 100
A1 = 1
A2 = 0.5

# sin波
x = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)

# FFT
X = np.fft.fft(x)

# 周波数成分
freq = np.fft.fftfreq(len(x), d=1/fs)
idx = np.argsort(freq)
# subplots
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# 元のsin波
axs[0].plot(t, x)
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Amplitude')

# 周波数成分
#axs[1].stem(freq[idx], np.abs(X[idx]) * 2 / len(x))
axs[1].plot(freq[idx], np.abs(X[idx]) * 2 / len(x))
axs[1].set_xlabel('Frequency [Hz]')
axs[1].set_ylabel('Magnitude')
plt.show()
