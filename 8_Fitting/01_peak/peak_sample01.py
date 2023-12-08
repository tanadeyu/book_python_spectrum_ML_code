import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# 疑似的な信号を生成します
x = np.linspace(1000, 4000, 1000)
y = 0.75*np.exp(-(x - 1750)**2 / (2 * 100**2)) + \
    1.00*np.exp(-(x - 3200)**2 / (2 * 150**2)) + \
    np.random.normal(0, 0.075, len(x))


# フィッティングで使用する関数を定義します
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))


# ピークを指定しフィッティングを行います。初期値にはa,b,cの近い値を入れます
popt1, _ = curve_fit(gaussian, x, y, p0=[0.5, 2000, 100])
popt2, _ = curve_fit(gaussian, x, y, p0=[1.5, 3000, 150])


# ピーク位置の代入
peak1_pos, peak1_int = popt1[1], popt1[0]
peak2_pos, peak2_int = popt2[1], popt2[0]

#print
peak1x='{:.0f}'.format(peak1_pos)
peak1y='{:.2f}'.format(peak1_int)
peak2x='{:.0f}'.format(peak2_pos)
peak2y='{:.2f}'.format(peak2_int)

# グラフをグレースケールで表示します
plt.plot(x, y,label="Original",color='0.75')
plt.plot(x, gaussian(x, *popt1),linewidth=3,label="fit1",linestyle="dashed",color='0.3')
plt.plot(x, gaussian(x, *popt2),linewidth=3,label="fit2",linestyle="dotted",color='0.0')

# フィッティングで求められたピークを表示します
plt.annotate('peak1: (' + peak1x + "," + peak1y + ")", \
             xy=(peak1_pos, peak1_int+0.05), xytext=(peak1_pos, peak1_int+0.25), \
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('peak2: (' + peak2x + "," + peak2y + ")", \
             xy=(peak2_pos, peak2_int+0.05), xytext=(peak2_pos, peak2_int+0.25), \
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel("Wavenumber (cm^-1)")
plt.ylabel("Intensity")
plt.ylim(0., 1.75) 
plt.legend()
plt.tight_layout()
plt.show()



