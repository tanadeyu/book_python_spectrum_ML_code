import math

#　リストの作成
x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]
n = len(x)

# 相関係数の計算
mean_x = sum(x) / n
mean_y = sum(y) / n
cov_xy = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(n)]) /n
std_x  = math.sqrt(sum([(x[i] - mean_x) ** 2 for i in range(n)]) /n)
std_y  = math.sqrt(sum([(y[i] - mean_y) ** 2 for i in range(n)]) /n)
coef   = cov_xy / (std_x * std_y)
print(coef)
