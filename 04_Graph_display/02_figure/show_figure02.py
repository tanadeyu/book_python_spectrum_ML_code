import matplotlib.pyplot as plt

# データの準備
x = [1, 2, 3, 4, 5]
y1 = [2, 3, 4, 1, 5]
y2 = [2, 4, 6, 8, 10]
# グラフを作成
fig, axs = plt.subplots(2, sharex=True)  # 2つのサブプロットを持つ。X軸を共有する

# サブプロット1にプロット
axs[0].plot(x, y1, color='b', marker='o', linestyle='-', linewidth=2, markersize=8)
axs[0].set_ylabel('Y-axis 1')

# サブプロット2にプロット
axs[1].plot(x, y2, color='g', marker='s', linestyle='--', linewidth=2, markersize=8)
axs[1].set_xlabel('X-axis')
axs[1].set_ylabel('Y-axis 2')

# グラフを表示
plt.tight_layout()
plt.show()
