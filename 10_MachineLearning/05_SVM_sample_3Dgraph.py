import matplotlib.pyplot as plt
# matplotlib日本語表示
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'
from sklearn import datasets, model_selection, svm, metrics

# データセットの読み込みと、トレーニングデータとテストデータの分割
iris = datasets.load_iris()
X, y = iris.data[:, :], iris.target
X_train, X_test, y_train, y_test = \
model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
jp_feature_names = \
['がく片の長さ [cm]', 'がく片の幅 [cm]', '花びらの長さ [cm]', '花びらの幅 [cm]']
jp_target_names = ['セトサ種', 'バージカラー種', 'バージニカ種']

# モデルの決定と学習
clf = svm.SVC(kernel='poly').fit(X_train, y_train)

#　推定値の計算
y_pred = clf.predict(X)

# Accuracyの計算と表示
accuracy = metrics.accuracy_score(y_test, clf.predict(X_test))

# 散布図作成の準備(軸の設定)
expv = [2,3,0] # 説明変数を3つ選択
common_xlim = X[:, expv[0]].min()-0.5, X[:, expv[0]].max()+0.5
common_ylim = X[:, expv[1]].min()-0.5, X[:, expv[1]].max()+0.5
common_zlim = X[:, expv[2]].min()-0.5, X[:, expv[2]].max()+0.5

# 散布図の表示
fig, (ax1, ax2) \
= plt.subplots(1, 2, subplot_kw={'projection':'3d'}, figsize=(10, 5))
colors = ['r', 'g', 'b']
markers = ['o', '^', ',']
for ax, x_data, y_data in zip([ax1, ax2], [X, X], [y, y_pred]):
    for i, color, marker in zip(clf.classes_, colors, markers):
        ax.scatter(x_data[y_data == i, expv[0]], x_data[y_data == i, expv[1]], \
        x_data[y_data == i, expv[2]], c=color, marker=marker, s=50,
        label=jp_target_names[i])
    ax.set_xlabel(jp_feature_names[expv[0]], fontsize=14)
    ax.set_ylabel(jp_feature_names[expv[1]], fontsize=14)
    ax.set_zlabel(jp_feature_names[expv[2]], fontsize=14)
    ax.set_xlim(common_xlim)
    ax.set_ylim(common_ylim)
    ax.set_zlim(common_zlim)
ax1.set_title("オリジナルデータ")
ax2.set_title('推定データ (Accuracy: {:.3f})'.format(accuracy), fontsize=14)
plt.legend(loc='best')
plt.show()