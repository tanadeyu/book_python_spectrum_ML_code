import matplotlib.pyplot as plt
from sklearn import datasets, model_selection, svm, metrics

# データセットの読み込みと、トレーニングデータとテストデータの分割
iris = datasets.load_iris()
X, y = iris.data[:, :3], iris.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(\
                                   X, y, test_size=0.5, random_state=0)

# モデルの決定と学習
clf = svm.SVC(kernel='poly').fit(X_train, y_train)

# 散布図の表示
fig, (ax1, ax2) \
= plt.subplots(2, 1, subplot_kw={'projection':'3d'}, figsize=(5, 10))
colors = ['r', 'g', 'b']
markers = ['o', '^', ',']
for ax, data in zip([ax1, ax2], [X_train, X_test]):
    y_pred = clf.predict(data)
    for i, color, marker in zip(clf.classes_, colors, markers):
        ax.scatter(data[y_pred == i, 0], data[y_pred == i, 1], \
        data[y_pred == i, 2], c=color, marker=marker, s=50)
    ax.set_xlabel(iris.feature_names[0], fontsize=13)
    ax.set_ylabel(iris.feature_names[1], fontsize=13)
    ax.set_zlabel(iris.feature_names[2], fontsize=13)
common_xlim = (min(X_train[:, 0].min(), X_test[:, 0].min()), \
               max(X_train[:, 0].max(), X_test[:, 0].max()))
common_ylim = (min(X_train[:, 1].min(), X_test[:, 1].min()), \
               max(X_train[:, 1].max(), X_test[:, 1].max()))
common_zlim = (min(X_train[:, 2].min(), X_test[:, 2].min()), \
               max(X_train[:, 2].max(), X_test[:, 2].max()))
ax1.set_xlim(common_xlim)
ax1.set_ylim(common_ylim)
ax1.set_zlim(common_zlim)
ax1.set_title("X_train")
ax2.set_xlim(common_xlim)
ax2.set_ylim(common_ylim)
ax2.set_zlim(common_zlim)
ax2.set_title("X_test")

# Accuracyの計算と表示
accuracy = metrics.accuracy_score(y_test, clf.predict(X_test))
plt.suptitle('Accuracy: {:.2f}'.format(accuracy), fontsize=13)
plt.show()