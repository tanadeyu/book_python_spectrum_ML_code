import matplotlib.pyplot as plt
# matplotlib日本語表示
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import numpy as np
from scipy import stats

# ワインデータセットをロード
wine = datasets.load_wine()
X = wine.data
y = wine.target
fnames_jp = \
["アルコール度数", "リンゴ酸", "灰分", "灰分のアルカリ度", \
 "マグネシウム", "全フェノール含量", "フラボノイド", \
 "非フラボノイドフェノール", "プロアントシアニジン", "色の濃さ", \
 "色相", "蒸留ワインのOD280/OD315", "プロリン"]

# データをトレーニング用とテスト用に分割
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.5, random_state=42)

# ガウシアンナイーブベイズモデルを作成し、トレーニングデータにフィットさせる
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# モデルの精度を表示
print("精度:", gnb.score(X_test, y_test))
score_txt = "(精度: " + '{:.4f}'.format(gnb.score(X_test, y_test)) + ')'

# 予測値を計算
predicted = gnb.predict(X_test)

# オリジナルと推定値で散布図を作成(説明変数は0と1のIndexを使用)
fig, ax = plt.subplots(1, 2, figsize=(7, 4))  # 図を作成
colors = ['r', 'g', 'b']
markers = ['o', '^', ',']
titles = ['オリジナル', '推定'+score_txt]
dsets = [y_test, predicted]
for i in range(2):
    for j, color in enumerate(colors):
        ax[i].scatter(X_test[dsets[i] == j, 0], X_test[dsets[i] == j, 1], \
        marker=markers[j], label=wine.target_names[j], s=50,\
        facecolor='None', edgecolors=color)
        ax[i].set_xlabel(fnames_jp[0])
        ax[i].set_ylabel(fnames_jp[1])
        ax[i].legend(loc='best')
        ax[i].set_title(titles[i])
plt.tight_layout()
plt.show()

# 確率密度の図を作成
fig, ax = plt.subplots(2, 3, figsize=(9, 5))  # 図を作成
for i in range(2):
  for j in range(3):
    x = np.linspace(X_test[:, i].min(), X_test[:, i].max(), 100)
    y = stats.norm.pdf(x, gnb.theta_[j, i], np.sqrt(gnb.var_[j, i]))
    #area = np.trapz(y, x);print(area)
    #y_normalized = y / area
    ax[i][j].plot(x, y)
    ax[i][j].set_title(wine.target_names[j])
    ax[i][j].set_xlabel(fnames_jp[i])
    ax[i][j].set_ylabel('確率密度')
plt.tight_layout()
plt.show()