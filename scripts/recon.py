# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# %%
root = Path.home() / "Downloads/135245/135245"
path = root / "NormalUS4.bin"
rf = np.fromfile(path, dtype=np.dtype(">d"), sep="")
rf = rf.reshape((1000, 2 * 3200))

# %%
plt.plot(rf[0])


# %%
# FIR filter
def apply_fir_filt(x, kernel):
    x_filt = np.empty_like(x)
    for i in range(x.shape[0]):
        x_filt[i] = np.convolve(x[i], kernel, "same")
    return x_filt


kernel = signal.firwin2(65, freq=[0, 0.1, 0.3, 1], gain=[0, 1, 1, 0])
rf_filt = apply_fir_filt(rf, kernel)


# %%
# Envelope detection
rf_env = np.abs(signal.hilbert(rf_filt))


# %%
# Log compression
def log_compress(x, db):
    x = x / x.max()
    x_log = 255 * (20 / db * np.log10(x) + 1)
    x_log = np.where(x_log < 1, 0, x_log)
    return x_log


rf_log = log_compress(rf_env, 45)
plt.imshow(rf_log.T, "gray", extent=[-1, 1, -1, 1])
