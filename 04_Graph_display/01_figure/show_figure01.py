import matplotlib.pyplot as plt

# データの準備
x = [1, 2, 3, 4, 5]
y = [2, 3, 7, 1, 5]

# グラフを作成
fig, ax = plt.subplots()
ax.plot(x, y, color='b', marker='o', linestyle='-', linewidth=2, markersize=8)

# タイトルとラベルを設定
ax.set_title('Single Graph')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# グラフを表示
plt.show()
