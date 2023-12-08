import numpy as np
import matplotlib.pyplot as plt

# サンプリング周波数
fs = 1000
# 時間軸
t = np.arange(0, 1, 1/fs)

# 3つのcos波を生成
f = 100
x1 = 1*np.cos(2*np.pi*f*t)
x2 = 0.75*np.cos(2*np.pi*f*t + np.pi/2)
x3 = 1.5*np.cos(2*np.pi*f*t - np.pi/2)

# フーリエ変換
X1 = np.fft.fft(x1)
X2 = np.fft.fft(x2)
X3 = np.fft.fft(x3)

# 振幅スペクトル
amp1 = np.abs(X1)
amp2 = np.abs(X2)
amp3 = np.abs(X3)

# 正規化した振幅スペクトル
amp1_norm = amp1 / fs *2
amp2_norm = amp2 / fs *2
amp3_norm = amp3 / fs *2

# 位相スペクトル
phase1 = np.angle(X1, deg=False)
phase2 = np.angle(X2, deg=False)
phase3 = np.angle(X3, deg=False)

# arctanで位相を-180度から180度の範囲に変換
phase1 = np.arctan2(np.sin(phase1), np.cos(phase1))/np.pi*180
phase2 = np.arctan2(np.sin(phase2), np.cos(phase2))/np.pi*180
phase3 = np.arctan2(np.sin(phase3), np.cos(phase3))/np.pi*180


# グラフを描画
fig, axs = plt.subplots(4, 3, figsize=(7, 7))


# オリジナルデータ
axs[0, 0].plot(t, x1)
axs[0, 0].set_title('Data 1')
axs[0, 0].set_xlim(0.0,0.1)
axs[0, 0].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 1].plot(t, x2)
axs[0, 1].set_title('Data 2')
axs[0, 1].set_xlim(0.0,0.1)
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('Amplitude')
axs[0, 2].plot(t, x3)
axs[0, 2].set_title('Data 3')
axs[0, 2].set_xlim(0.0,0.1)
axs[0, 2].set_xlabel('Time [s]')
axs[0, 2].set_ylabel('Amplitude')


# 正規化した振幅
axs[1, 0].plot(amp1_norm)
axs[1, 0].set_xlabel('Frequency [Hz]')
axs[1, 0].set_ylabel('Magnitude')
axs[1, 0].set_xlim(0,500)
axs[1, 1].plot(amp2_norm)
axs[1, 1].set_xlabel('Frequency [Hz]')
axs[1, 1].set_ylabel('Magnitude')
axs[1, 1].set_xlim(0,500)
axs[1, 2].plot(amp3_norm)
axs[1, 2].set_xlabel('Frequency [Hz]')
axs[1, 2].set_ylabel('Magnitude')
axs[1, 2].set_xlim(0,500)


# 位相
axs[2, 0].plot(phase1)
axs[2, 0].set_xlabel('Frequency [Hz]')
axs[2, 0].set_ylabel('Phase[deg]')
axs[2, 0].set_xlim(0,500)
axs[2, 1].plot(phase2)
axs[2, 1].set_xlabel('Frequency [Hz]')
axs[2, 1].set_ylabel('Phase[deg]')
axs[2, 1].set_xlim(0,500)
axs[2, 2].plot(phase3)
axs[2, 2].set_xlabel('Frequency [Hz]')
axs[2, 2].set_ylabel('Phase[deg]')
axs[2, 2].set_xlim(0,500)


# 位相2
axs[3, 0].plot(phase1)
axs[3, 0].set_xlabel('Frequency [Hz]')
axs[3, 0].set_ylabel('Phase[deg]')
axs[3, 0].set_xlim(98,102)
axs[3, 0].set_ylim(-180,180)
axs[3, 1].plot(phase2)
axs[3, 1].set_xlabel('Frequency [Hz]')
axs[3, 1].set_ylabel('Phase[deg]')
axs[3, 1].set_xlim(98,102)
axs[3, 1].set_ylim(-180,180)
axs[3, 2].plot(phase3)
axs[3, 2].set_xlabel('Frequency [Hz]')
axs[3, 2].set_ylabel('Phase[deg]')
axs[3, 2].set_xlim(98,102)
axs[3, 2].set_ylim(-180,180)


#　100Hzでの位相の表示
print('phase1@100Hz[degree] =',phase1[100])
print('phase2@100Hz[degree] =',phase2[100])
print('phase3@100Hz[degree] =',phase3[100])


#　グラフ化
plt.tight_layout()
plt.show()
