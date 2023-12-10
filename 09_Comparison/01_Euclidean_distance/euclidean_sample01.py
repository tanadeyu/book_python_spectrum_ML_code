from math import sqrt

# 2つの信号の値
x1, x2, x3 = 1,  2, 3
y1, y2, y3 = 5, -1, 3

# ユークリッド距離を計算
distance = sqrt((x1 - y1)**2 + (x2 - y2)**2 + (x3 - y3)**2)

# 結果を表示
print(distance)
