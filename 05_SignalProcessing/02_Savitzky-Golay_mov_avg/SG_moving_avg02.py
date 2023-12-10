import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ガウシアンから疑似的なスペクトル信号を生成
x = np.linspace(400, 800, 1000)  # 波長の範囲
gaussian_signal = 32000 * np.exp(-(x - 520)**2 / (2 * 3.0**2))  # ガウシアンスペクトル信号
noise = np.random.normal(0, 1000, gaussian_signal.shape)
gaussian_signal= gaussian_signal+noise

# 移動平均を計算
window_size = 21  # 移動平均ウィンドウサイズ（奇数）
smoothed_signal = savgol_filter(gaussian_signal, window_size, 2)  # savitzky-golayフィルタを使用した平滑化

# グラフ化
plt.figure(figsize=(10, 6))
plt.plot(x, gaussian_signal, label='Original Signal', color='blue', alpha = 0.3)
plt.plot(x, smoothed_signal, label='Smoothed Signal', color='red',linewidth=3,linestyle='--')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Spectrum Signal and Moving Average')
plt.legend()
plt.grid(True)
plt.show()
