import matplotlib.pyplot as plt

# matplotlib日本語表示
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

jp_feature_names = \
    ['がく片の長さ [cm]', 'がく片の幅 [cm]', '花びらの長さ [cm]', '花びらの幅 [cm]']
jp_target_names = ['セトサ種', 'バージカラー種', 'バージニカ種']


# データの読み込み
iris = load_iris()
X = iris.data[:, 2:]  # 花弁の長さと幅のみを使用
y = iris.target

# 決定木モデルの作成
tree_clf = DecisionTreeClassifier(max_depth=3, criterion="gini",random_state=10)
tree_clf.fit(X, y)

# ジニ不純度値の計算
def gini(p):
    return 1 - (p ** 2).sum(axis=1)

# ジニ不純度値のプロット
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = tree_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.1, cmap="gist_ncar")

# クラスごとにマーカーの形を指定して散布図をプロット
for i, marker, color in zip(range(3), ['o', '^', ','], ['r', 'g', 'b']):
    plt.scatter(X[y == i, 0], X[y == i, 1], marker=marker, s=40, c=color, \
                label=jp_target_names[i],facecolor='None')

plt.xlabel(jp_feature_names[2])
plt.ylabel(jp_feature_names[3])

plt.legend(loc='best')
plt.show()

# ジニ不純度値の表示
gini_values = gini(tree_clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]))
gini_values = gini_values.reshape(xx.shape)
plt.contourf(xx, yy, gini_values, alpha=0.1, cmap="gist_ncar")
plt.colorbar()
plt.show()
