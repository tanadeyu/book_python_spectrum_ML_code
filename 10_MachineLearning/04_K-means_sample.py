
import os
os.environ["OMP_NUM_THREADS"] = "1"
value = os.environ["OMP_NUM_THREADS"]
print(value)
#import warnings
#warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
# matplotlib日本語表示
plt.rcParams['font.family'] = 'Noto Sans JP'
#plt.rcParams['font.family'] = 'Noto Serif JP'
#plt.rcParams['font.family'] = 'Yu Gothic'
#plt.rcParams['font.family'] = 'MS Gothic'


from sklearn import datasets
from sklearn.cluster import KMeans

# irisデータセットを読み込む
iris = datasets.load_iris()
X = iris.data

# k-means法でクラスタリング
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

jp_feature_names = \
    ['がく片の長さ [cm]', 'がく片の幅 [cm]', '花びらの長さ [cm]', '花びらの幅 [cm]']
#教師なしのため、対応は不明
#jp_target_names = ['セトサ種', 'バージカラー種', 'バージニカ種']


# クラスタごとにプロット
exp_var1, exp_var2 = 0, 1 #説明変数0,1番目を使う
plt.figure(figsize=(5, 5))
markers = ['o', '^', 's']
colors = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(X[y_kmeans == i, exp_var1], X[y_kmeans == i, exp_var2], \
    s=50, c=colors[i], marker=markers[i], label=f'Cluster {i}:')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, exp_var1], centers[:, exp_var2], \
s=300, c='black', marker='*', label='Centroids') # クラスタの中心をプロット

# グラフを表示
plt.legend()
plt.xlabel(jp_feature_names[exp_var1])
plt.ylabel(jp_feature_names[exp_var2])
plt.tight_layout()
plt.show()


# クラスタごとにプロット
exp_var1, exp_var2 = 2, 3 #説明変数2,3番目を使う
plt.figure(figsize=(5, 5))
markers = ['o', '^', 's']
colors = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(X[y_kmeans == i, exp_var1], X[y_kmeans == i, exp_var2], \
    s=50, c=colors[i], marker=markers[i], label=f'Cluster {i}:')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, exp_var1], centers[:, exp_var2], \
s=300, c='black', marker='*', label='Centroids') # クラスタの中心をプロット

# グラフを表示
plt.legend()
plt.xlabel(jp_feature_names[exp_var1])
plt.ylabel(jp_feature_names[exp_var2])
plt.tight_layout()
plt.show()