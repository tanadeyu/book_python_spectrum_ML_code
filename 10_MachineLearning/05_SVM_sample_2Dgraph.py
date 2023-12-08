import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Import train_test_split

# Irisデータセットをロード
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # 3番目と4番目の説明変数を取得
y = iris.target

# 標準化を行う
scaler = StandardScaler()
X = scaler.fit_transform(X)

# トレーニングデータとテストデータを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# SVMモデルを構築
classifiers = [
    SVC(kernel="linear",  C=3.0),
    SVC(kernel="poly",    degree=3, C=3.0),
    SVC(kernel="rbf",     gamma=0.5, C=3.0),
    SVC(kernel="sigmoid", gamma=0.5, C=3.0)
]

# グラフの描画
fig, sub = plt.subplots(2, 2,figsize=(8, 6))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# 各モデルで学習と描画 (clfはclassification)
for clf, title, ax  in zip(classifiers, \
    ['Linear SVM', 'Poly SVM', 'RBF SVM', 'Sigmoid SVM'], sub.flatten()):
    clf.fit(X_train, y_train)
    # 散布図の描画
    for i, color, marker in zip(clf.classes_, ['r', 'g', 'b'], \
        ['o', '^', ',']):
        idx = np.where(y_train == i)
        ax.scatter(X_train[idx, 0], X_train[idx, 1], c=color, \
        label=iris.target_names[i], \
        marker=marker, edgecolor='k', s=60)
    # 境界線の描画
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), \
    np.linspace(ylim[0], ylim[1], 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pltt=ax.contourf(xx, yy, Z, alpha=0.15, cmap=plt.cm.binary)
    ax.set_xlabel('Feature 3')
    ax.set_ylabel('Feature 4')
    ax.set_title(title)
    fig.colorbar(pltt,ax=ax)

    # サポートベクターに丸を付けて表示
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], \
    s=300, facecolors='none', edgecolors='red')

    # accuracyの計算と表示
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    ax.text(0.95, 0.05, ('Accuracy = %.2f' % acc).lstrip('0'), \
    size=12, ha='right', transform=ax.transAxes, color='black')

plt.tight_layout()
plt.show()