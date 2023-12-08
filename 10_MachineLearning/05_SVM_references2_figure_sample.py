import numpy as np
import matplotlib.pyplot as plt
# matplotlib日本語表示
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'

from sklearn import datasets
from sklearn import svm

# ワインのデータセットをロード
wine = datasets.load_wine()

# 0番目と6番目の説明変数を抽出
X = wine.data[:, [0, 6]]

# ターゲットを抽出（ターゲット変数0を1に、1，2を0にセット）
y = (wine.target == 0).astype(int)

# SVCモデルを作成
clf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)

# モデルを訓練
clf.fit(X, y)

# グラフのサイズを設定
plt.figure(figsize=(6, 4))

# 散布図をプロット
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], label='class 0', marker='o', s=50)
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], label='class 0以外', marker='^', s=50)

# サポートベクターをプロット
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150, \
            facecolors='none', edgecolors='k', label='サポートベクター')

# 決定境界をプロット
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1.5
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # マージンの計算
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])  # マージンを含む決定境界

# 軸ラベルを設定
plt.xlabel('0番目の特徴量')
plt.ylabel('6番目の特従量')

# グラフを表示
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()