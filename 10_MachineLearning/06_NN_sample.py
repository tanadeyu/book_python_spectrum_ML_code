import matplotlib.pyplot as plt
# matplotlib日本語表示
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ワインのデータセットをロード
data = load_wine()
X = data.data
y = data.target

feature_names_jp = \
["アルコール度数", "リンゴ酸", "灰分", "灰分のアルカリ度", \
 "マグネシウム", "全フェノール含量", "フラボノイド", \
 "非フラボノイドフェノール", "プロアントシアニジン", "色の濃さ", \
 "色相", "蒸留ワインのOD280/OD315", "プロリン"]


# 0番目と6番目の特徴量を抽出
X = X[:, [0, 6]]

# データの正規化
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.4, random_state=0)

# ニューラルネットワークモデルの構築
model = MLPClassifier(hidden_layer_sizes=(10, 10), \
                      activation='relu', solver='sgd', \
                      max_iter=5000, random_state=0,tol=1e-5)

# モデルの訓練
history = model.fit(X_train, y_train)

# テストデータの予測
y_pred = model.predict(X_test)

# 正確性を計算
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# 散布図を表示
plt.figure(figsize=(8, 4))

# テストに振り分けた元データの散布図を表示
plt.subplot(1, 2, 1)
colors = ['r', 'g', 'b']
markers = ['o', '^', ',']
for i in range(3):
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1], \
    c=colors[i], marker=markers[i], label=f'Class {i}')
plt.title('元データ')
plt.xlabel(feature_names_jp[0])
plt.ylabel(feature_names_jp[6])
plt.legend(loc='upper right')

# テストの説明変数から推定した散布図を表示
plt.subplot(1, 2, 2)
for i in range(3):
    plt.scatter(X_test[y_pred == i, 0], X_test[y_pred == i, 1], \
    c=colors[i], marker=markers[i], label=f'Class {i}')
plt.title('推定結果')
plt.xlabel(feature_names_jp[0])
plt.ylabel(feature_names_jp[6])
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()

# 試行回数と損失値を表示
plt.figure(figsize=(4, 4))
plt.plot(history.loss_curve_)
plt.title('損失カーブ')
plt.xlabel('試行回数')
plt.ylabel('損失')
plt.tight_layout()
plt.show()