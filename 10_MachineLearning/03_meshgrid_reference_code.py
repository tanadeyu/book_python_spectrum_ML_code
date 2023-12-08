
import numpy as np
X, Y = np.meshgrid(np.arange(3), np.arange(3))
print("X\n",X)
print("Y\n",Y)
print("\n")
print("X.ravel()\n",X.ravel())
print("Y.ravel()\n",Y.ravel())
print("\n",'np.c_[X.ravel(), Y.ravel()]')
print(np.c_[X.ravel(), Y.ravel()])
