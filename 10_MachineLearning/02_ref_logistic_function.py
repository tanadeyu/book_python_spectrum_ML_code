
import numpy as np
import matplotlib.pyplot as plt

def logistic_function(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-6, 6, 100)  # -6から6までの範囲で100点を生成
y = logistic_function(x)

plt.plot(x, y)
plt.title('Logistic Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()