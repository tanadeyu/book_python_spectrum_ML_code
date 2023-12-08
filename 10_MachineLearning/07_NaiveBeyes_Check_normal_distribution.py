import matplotlib.pyplot as plt
# matplotlib日本語表示
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'

import numpy as np
from scipy import stats
from sklearn.datasets import load_wine

# ワインのデータセットをロード
data = load_wine()
X = data.data
fnames_jp = \
["アルコール度数", "リンゴ酸", "灰分", "灰分のアルカリ度", \
 "マグネシウム", "全フェノール含量", "フラボノイド", \
 "非フラボノイドフェノール", "プロアントシアニジン", "色の濃さ", \
 "色相", "蒸留ワインのOD280/OD315", "プロリン"]


# 説明変数分の13個のサブプロット分を確保
fig, axs = plt.subplots(4, 4, figsize=(10, 10))
axs = axs.ravel()

# 各特徴量についてヒストグラムを作成
for i in range(X.shape[1]):
    ax = axs[i]
    feature = X[:, i]
    ax.hist(feature, bins=15, density=True, alpha=0.9, color='w', histtype="bar",edgecolor="black")

    # 正規分布の直線を追加
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(feature), np.std(feature))
    ax.plot(x, p, 'k', linewidth=2)

    # Shapiro-Wilk検定のp値と統計量を追加
    stat, p_value = stats.shapiro(feature)
    ax.text(0.05, 0.8, f'P値: {p_value:.3f}\n統計量: {stat:.3f}', transform=ax.transAxes)

    #ax.set_title(data.feature_names[i])
    ax.set_title(fnames_jp[i])

plt.tight_layout()
plt.show()