import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])
coef = np.corrcoef(x, y)
print(coef)
