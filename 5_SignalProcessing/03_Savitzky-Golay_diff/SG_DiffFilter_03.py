import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# ガウシアン信号の生成
np.random.seed(0)
x = np.linspace(0, 10, 100)
gaussian_signal = np.exp(-(x - 5)**2 / (2 * 1.5**2)) + np.random.normal(0, 0.01, len(x))
# Savitzky-Golayフィルタを使用した微分
window_size = 9  # ウィンドウサイズ（奇数を指定すると良い）
order = 1  # 多項式の次数（1次微分を行うので1）
derivative = savgol_filter(gaussian_signal, window_size, polyorder=order, deriv=1)

# グラフの描画
plt.figure(figsize=(12, 6))

# 元データのプロット
plt.subplot(2, 1, 1)
plt.plot(x, gaussian_signal, color='b', label='Original Signal')
plt.title('Original Gaussian Signal')
plt.xlabel('X')
plt.ylabel('Amplitude')
plt.legend()

# Savitzky-Golay微分後のデータのプロット
plt.subplot(2, 1, 2)
plt.plot(x, derivative, color='r', label='Derivative (Savitzky-Golay)')
plt.title('Derivative using Savitzky-Golay Filter')
plt.xlabel('X')
plt.ylabel('Amplitude')
plt.legend()


plt.tight_layout()
plt.show()
