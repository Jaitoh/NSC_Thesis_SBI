import numpy as np

# Posterior probabilities from input1
posterior1 = np.array([0.2, 0.8])

# Posterior probabilities from input2
posterior2 = np.array([0.6, 0.4])

# Assuming independence, combine by multiplying
combined_posterior = posterior1 * posterior2

# Normalize so probabilities sum to 1
normalized_posterior = combined_posterior / combined_posterior.sum()

# print(normalized_posterior)

import numpy as np
import matplotlib.pyplot as plt

# 给定的信号
signal = np.array([-1, 0, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1])*0.2
signal = np.array([-1, 1, 1, -1, 0, 0 , 0, 0, 0, 0 , 0 , 0 , 0, 0 , 0])*0.2

# integrate the signal
integrated_signal = np.cumsum(signal)
    
# 使用FFT计算频域成分
X = np.fft.fft(integrated_signal)

# 计算幅度
magnitude = np.abs(X)

# 计算频率（在这种情况下，是归一化的频率，单位是周期/采样）
frequencies = np.fft.fftfreq(len(signal))

# 画出频谱
plt.stem(frequencies, magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum")
plt.show()

plt.plot(integrated_signal, '.-')
plt.plot(signal, '.-')
