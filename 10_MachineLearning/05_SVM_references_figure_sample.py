import numpy as np
import matplotlib.pyplot as plt

# matplotlib日本語表示
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'

from sklearn import datasets
from sklearn import svm

# Irisデータセットをロード
iris = datasets.load_iris()

# 3番目と4番目の説明変数を抽出
X = iris.data[iris.target!=2]
X = X[:, 2:4]
# ターゲットを抽出
y = iris.target[iris.target!=2]

# SVCモデルを作成
clf = svm.SVC(kernel='linear')

# モデルを訓練
clf.fit(X, y)

# グラフのサイズを設定
plt.figure(figsize=(6, 4))

# 散布図をプロット
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], label='ターゲット 0', marker='o',s=50)
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], label='ターゲット 1', marker='^',s=50)

# サポートベクターをプロット
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150, \
            facecolors='none', edgecolors='k', label='サポートベクター')

# マージンとその値を表示
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
plt.text(2.5, 2, f'マージンの値: {margin:.2f}', fontsize=12)

# 決定境界をプロット
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], \
            linestyles=['--', '-', '--'], levels=[-margin, 0, margin])

# 軸ラベルを設定
plt.xlabel('花びらの長さ')
plt.ylabel('花びらの幅')

# グラフを表示
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()