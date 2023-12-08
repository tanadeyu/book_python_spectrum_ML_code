import matplotlib.pyplot as plt

# matplotlib日本語表示
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'

# 必要なライブラリをインポート
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Wineデータセットを読み込む
wine = load_wine()

# データフレームに変換する
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# データ項目を日本語に置き換え
feature_names_jp = \
["アルコール度数", "リンゴ酸", "灰分", "灰分のアルカリ度", \
 "マグネシウム", "全フェノール含量", "フラボノイド", \
 "非フラボノイドフェノール", "プロアントシアニジン", "色の濃さ", \
 "色相", "蒸留ワインのOD280/OD315", "プロリン"]
df.columns = feature_names_jp

# データを学習用とテスト用に分割する
X_train, X_test, y_train, y_test \
= train_test_split(df, wine.target, test_size=0.3, random_state=0)

# 正規化(ロジスティック回帰のモデル精度向上のために行う)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# ロジスティック回帰モデルを学習する（Cは初期値の1.0とする）
lr = LogisticRegression(C=1.0, random_state=0)
lr.fit(X_train_std, y_train)

# テストデータで予測を行い、正解率を計算する
y_pred = lr.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print('正解率: %.2f' % accuracy)

# 学習したパラメータを表示
print(lr.coef_)  # 係数
print(lr.intercept_)  # 切片

# グラフとラベルを使って表示
plt.figure()
colors = ['r', 'g', 'b']
markers = ['o', '^', ","]
lw = 2

# 例として説明変数0アルコール度数と、6フラボノイドで
# ワインのクラスを散布図でグラフ化
for mtemp, ctmp, i, target_name \
in zip(markers, colors, [0, 1, 2], wine.target_names):
    plt.scatter(wine.data[wine.target == i, 0], \
    wine.data[wine.target == i, 6], \
    marker=mtemp, color=ctmp, alpha=.8, lw=lw, \
    label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Wine dataset')
plt.show()

