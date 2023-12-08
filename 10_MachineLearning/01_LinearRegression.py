import pandas as pd
import matplotlib.pyplot as plt

# matplotlib日本語表示
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
 
# カリフォルニア住宅価格データセットの読み込み
# Xに説明変数、yにターゲット（住宅価格）を代入
california_housing = fetch_california_housing()
X, y = california_housing.data, california_housing.target
feature_names = california_housing.feature_names
 
# Xとyのデータを一度まとめる（Pandas DataFrameに変換）
data = pd.DataFrame(data=X, columns=feature_names)
data['Target'] = y
 
# データの内容を一部表示
print("Data:")
print(data.head())
 
# あらためて説明変数とターゲット変数に分ける
# （Target以外をターゲット変数に指定することも可能）
X = data.drop('Target', axis=1)
y = data['Target']
 
# データの分割(トレーニング用80％、テスト用20％)
# モデル再現性確認のためrandom_stateを指定
X_train, X_test, y_train, y_test = train_test_split(X, y, \
  test_size=0.2, random_state=42)
 
# モデルの選択と訓練
model = LinearRegression()
model.fit(X_train, y_train)
 
# テストデータでの予測
y_pred = model.predict(X_test)
 
# モデルの評価(平均二乗誤差)
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
bias = model.intercept_
print("Bias:", bias)
 
# 予測値と実際の値のプロット
fig = plt.figure(figsize=(3, 6))
ax1 = fig.add_subplot(111)
plt.scatter(y_test, y_pred, color='blue',s=3)
plt.xlabel("実際の価格")
plt.ylabel("予測価格")
plt.title("実際の価格と予測価格")
ax1.set_aspect('equal')
#plt.tight_layout()
plt.show()

# 重みの表示
feature_names_jp = ['世帯所得', '築年数', '部屋数平均', \
                    '寝室数平均', '居住人数', '世帯人数平均', \
                    '代表地区緯度', '代表地区経度']
fig2 = plt.figure(figsize=(6, 3))
#coeff = pd.Series(model.coef_, index=california_housing.feature_names)
coeff = pd.Series(model.coef_, index=feature_names_jp)
coeff.plot(kind='bar')
plt.tight_layout()
plt.show()
