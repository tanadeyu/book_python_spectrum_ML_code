import numpy as np
import lmfit
import matplotlib.pyplot as plt

# ガウス関数の定義
def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

# x値の作成
x = np.linspace(-5, 5, 100)

# 信号を作成
signal1 = gaussian(x, 1,  -1, 0.5) +  np.random.normal(size=x.size, scale=0.15)
signal2 = gaussian(x, 2.0, 1.5, 1)  + np.random.normal(size=x.size, scale=0.15)
data = signal1 + signal2

# 変化するパラメータと初期値の設定
params = lmfit.Parameters()
params.add('amp1', value=1)
params.add('cen1', value=-1)
params.add('sigma1', value=1)
params.add('amp2', value=1)
params.add('cen2', value=1.5)
params.add('sigma2', value=1)

# 差の計算
def residual(params, x, data):
    amp1 = params['amp1'].value
    cen1 = params['cen1'].value
    sigma1 = params['sigma1'].value
    amp2 = params['amp2'].value
    cen2 = params['cen2'].value
    sigma2 = params['sigma2'].value
    model = gaussian(x, amp1, cen1, sigma1) + gaussian(x, amp2, cen2, sigma2)
    return model - data

# 差を最小化
result = lmfit.minimize(residual, params, args=(x, data))

# グラフをプロット
plt.plot(x, data, 'b', label='original signal',linestyle="dashed")
plt.plot(x, residual(result.params, x, data) + data, 'r', label='fit')
plt.legend()
plt.show()

# フィットしたパラメータを表示する
lmfit.report_fit(result.params)
