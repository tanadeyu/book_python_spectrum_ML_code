import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# スペクトルデータの準備（例）
wavelengths = np.arange(400, 1000, 10)  # 400から1000までの間隔10の波長データ（例）
intensity = np.random.rand(len(wavelengths))  # ダミーの強度データ（例）
# 移動平均のウィンドウサイズを定義
window_size = 5
# DataFrameにデータを格納
data = pd.DataFrame({'Wavelength': wavelengths, 'Intensity': intensity})
# 移動平均を計算
data['Moving Average'] = data['Intensity'].rolling(window=window_size, min_periods=1).mean()
# グラフを描画
plt.figure(figsize=(10, 6))
plt.plot(data['Wavelength'], data['Intensity'], label='Original Spectrum', color='b')
plt.plot(data['Wavelength'], data['Moving Average'], label='Moving Average', color='r', linewidth=3, linestyle='--')
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Spectrum with Moving Average')
plt.legend()
plt.show()
