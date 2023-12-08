import matplotlib.pyplot as plt
# matplotlib日本語表示
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'

import pandas as pd
#import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# irisデータセットを読み込む
iris = datasets.load_iris()
X = iris.data
y = iris.target

jp_feature_names = \
    ['がく片の長さ [cm]', 'がく片の幅 [cm]', '花びらの長さ [cm]', '花びらの幅 [cm]']
jp_target_names = ['セトサ種', 'バージカラー種', 'バージニカ種']

#散布図の表示
markers = ['o', '^', ","]
fig, axs = plt.subplots(4, 4, figsize=(10, 10))
for i in range(4):
  for j in range(4):
    if i != j:
      for target in range(3):
        axs[i, j].scatter(iris.data[iris.target == target, i],\
                          iris.data[iris.target == target, j],\
                          label=jp_target_names[target],\
                          marker=markers[target])
      axs[i, j].set_xlabel(jp_feature_names[i])
      axs[i, j].set_ylabel(jp_feature_names[j])
      axs[i, j].legend()
plt.tight_layout()
plt.show()

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)

# 決定木モデルを作成
dt_model = DecisionTreeClassifier(max_depth=2,random_state=0)
dt_model.fit(X_train, y_train)

# ランダムフォレストモデルを作成
rf_model = RandomForestClassifier(n_estimators=10, \
                                  max_depth=2,random_state=0)
rf_model.fit(X_train, y_train)

# モデルの正確性を出力
print("Decision Tree Accuracy:",   dt_model.score(X_test, y_test))
print("Random Forest Accuracy:",   rf_model.score(X_test, y_test))

# 特徴量の重要度(決定木)
feature_importances = dt_model.feature_importances_
feature_names = jp_feature_names
fig = plt.figure(figsize=(6, 3))
coeff = pd.Series(feature_importances, index=feature_names)
coeff.plot(kind='bar')
plt.title("特徴量の重要度(決定木)")
plt.tight_layout()
plt.show()

# 特徴量の重要度(ランダムフォレスト)
feature_importances = rf_model.feature_importances_
feature_names = jp_feature_names
fig = plt.figure(figsize=(6, 3))
coeff = pd.Series(feature_importances, index=feature_names)
coeff.plot(kind='bar')
plt.title("特徴量の重要度(ランダムフォレスト)")
plt.tight_layout()
plt.show()

# 決定木の可視化
plt.figure(figsize=(10,5))
plot_tree(dt_model, feature_names=jp_feature_names, \
          class_names=jp_target_names ,filled=False)
#plot_tree(dt_model, feature_names=iris.feature_names, \ 
         #class_names=iris.target_names.tolist() ,filled=False)
plt.tight_layout()
plt.show()

#　ランダムフォレスト[0]の可視化
plt.figure(figsize=(10,5))
plot_tree(rf_model.estimators_[0], feature_names=jp_feature_names, \
          class_names=jp_target_names ,filled=False)
plt.tight_layout()
plt.show()

#　ランダムフォレスト[1]の可視化
plt.figure(figsize=(10,5))
plot_tree(rf_model.estimators_[1], feature_names=jp_feature_names, \
          class_names=jp_target_names ,filled=False)
plt.tight_layout()
plt.show()

