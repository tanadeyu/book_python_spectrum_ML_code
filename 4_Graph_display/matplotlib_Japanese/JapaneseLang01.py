import matplotlib.pyplot as plt
import numpy as np

# 日本語フォントを指定
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'


# グラフの描画
x = np.linspace(-1, 1, 100)
y = np.exp(x)
plt.plot(x, y, label="指数関数", ls='--',lw=3)

# x=0の接線
plt.plot(0, 1, 'ro')  # x=0での点
plt.plot(x, 1 + x, label="x=0の接線",lw=3)

# グラフの装飾
plt.title("グラフタイトル")
plt.xlabel("x軸ラベル名")
plt.ylabel("y軸ラベル名")
plt.legend()

# グラフの表示
plt.tight_layout()
plt.show()