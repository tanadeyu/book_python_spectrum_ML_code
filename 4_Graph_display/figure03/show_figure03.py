import matplotlib.pyplot as plt
# データの準備
x = [1, 2, 3, 4, 5]
y1 = [1, 2, 3, 4, 5]
y2 = [1, 3, 2, 5, 4]
y3 = [5, 4, 3, 2, 1]
y4 = [4, 3, 3, 4, 1]

# グラフを作成
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)  # 2x2のサブプロットを持つ。X軸とY軸を共有する

# サブプロットにプロット
axs[0, 0].plot(x, y1, color='b', marker='o', linestyle='-', linewidth=2, markersize=8)
axs[0, 0].set_title('Graph 1')
axs[1, 0].plot(x, y2, color='r', marker='^', linestyle='-', linewidth=2, markersize=8)
axs[1, 0].set_title('Graph 2')
axs[0, 1].plot(x, y3, color='g', marker='s', linestyle='--', linewidth=2, markersize=8)
axs[0, 1].set_title('Graph 3')
axs[1, 1].plot(x, y4, color='m', marker='d', linestyle='--', linewidth=2, markersize=8)
axs[1, 1].set_title('Graph 4')

# グラフを表示
plt.tight_layout()
plt.show()
