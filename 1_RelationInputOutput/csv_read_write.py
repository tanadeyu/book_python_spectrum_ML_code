import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ガウス分布のパラメータを設定
mu, sigma = 0, 0.1


# x軸の値を生成
x = np.linspace(-1, 1, 100)


# ガウス分布の式を計算
y = norm.pdf(x, mu, sigma)


# グラフを描画
plt.plot(x, y)
plt.show()
