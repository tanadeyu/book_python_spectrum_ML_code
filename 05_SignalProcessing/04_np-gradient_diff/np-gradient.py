import numpy as np
import matplotlib.pyplot as plt

# ガウシアン信号の生成
np.random.seed(0)
x = np.linspace(0, 10, 101)
gaussian_signal = np.exp(-(x - 5) ** 2 / (2 * 1.5 ** 2)) + np.random.normal(0, 0.01, len(x))

# ガウシアン信号の微分（中心差分を使用）
dx = x[1] - x[0]
derivative = np.gradient(gaussian_signal, dx)

# グラフの描画
plt.figure(figsize=(12, 6))

# 元データのプロット
plt.subplot(2, 1, 1)
plt.plot(x, gaussian_signal, color='b', label='Original Signal')
plt.title('Original Gaussian Signal')
plt.xlabel('X')
plt.ylabel('Amplitude')
plt.legend()

# 微分後のデータのプロット
plt.subplot(2, 1, 2)
plt.plot(x, derivative, color='r', label='Derivative (Numerical)')
plt.title('Numerical Derivative')
plt.xlabel('X')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()
