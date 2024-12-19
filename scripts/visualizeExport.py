# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


path = Path.home() / "Desktop/20241030 invivo 270_ARPAM120745_40"
assert path.exists()

# %%
data = {}

USrf_p = path / "USrf.bin"
if USrf_p.exists():
    rf = np.fromfile(USrf_p, np.float32)
    rf = rf.reshape((1000, rf.size // 1000))

    data["USrf"] = rf

# %%
nrows = len(data)
ascan_i = 10
f, ax = plt.subplots(nrows, 1)

if isinstance(ax, np.ndarray):
    ax = ax.flatten()
else:
    ax = [ax]

for i, (k, arr) in enumerate(data.items()):
    ax[i].plot(arr[ascan_i])
    ax[i].set_title(k)

# %%
plt.plot(rf[100][20:-20])

# %%
rf[100].min()
