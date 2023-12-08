import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# 横軸の範囲を設定
x = np.linspace(4000, 400, 1000)

# ガウスの関数作成
def gaussian(x, mu, sigma, A):
    return A * np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2)))

# 信号を２つ生成します
signal1 = gaussian(x, 1500, 50, 1)
signal2 = gaussian(x, 1700, 100, 0.5)

# 2つの信号をグラフに描画します
plt.plot(x, signal1, label='Signal 1',lw=3)
plt.plot(x, signal2, label='Signal 2',lw=3,ls="dashed")
plt.xlabel('Wavenumber')
plt.ylabel('Intensity')
plt.legend()
plt.show()

# ユークリッド距離と類似度の結果を表示します
distance = euclidean(signal1, signal2)
print('Euclidean distance between the two signals:', distance)
print('Similarity:', 1/(1+distance))
