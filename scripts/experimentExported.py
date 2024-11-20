# %%
from pathlib import Path
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import cv2


root = Path.home() / "Desktop/ARPAM Seg Dev Data"
USenvs = [p / "USenv.bin" for p in root.glob("*")]
rfs = [np.fromfile(p, dtype=np.float32) for p in USenvs]
rfs = [rf.reshape(1000, rf.size // 1000) for rf in rfs]


# %%
def plot_rfs(rfs):
    n = len(rfs)
    f, ax = plt.subplots(n, 1, sharex=True, sharey=True)
    ax = ax.flatten()
    for i, rf in enumerate(rfs):
        ax[i].plot(rf[0])
        ax[i].set(ylim=[0, 1])


plot_rfs(rfs)


# %%
def log_compress(x, noise_floor, desired_dynamic_range_dB=45):
    # The noise floor should be constant for a given system and is predetermined
    # The desired DR after compression is used to scale

    # Determine the peak signal value.
    peak_level = np.max(x)
    dynamic_range_dB = 20 * np.log10(peak_level / noise_floor)
    # print(f"Dynamic range: {dynamic_range_dB} dB")

    # Apply log compression
    x_log = 20 * np.log10(x) - 20 * np.log10(noise_floor)
    # Normalize to the desired DR and cut off values below 0
    x_log = np.clip(x_log, 0, desired_dynamic_range_dB)
    # Scale to the range [0, 1]
    x_log /= desired_dynamic_range_dB
    return x_log, dynamic_range_dB


# %%
# %matplotlib qt
from scipy.ndimage import gaussian_filter
import numba as nb


def transform(rf):
    rf = gaussian_filter(rf, sigma=5)
    # rf = np.cumsum(rf)
    # rf = np.diff(rf, 1)
    return rf


@nb.njit()
def cusum_fast(signal_data, threshold_pos=0.4, drift=0.1, threshold_neg=0.2):
    """
    threshold = 0.4  # Adjust based on noise level and desired sensitivity
    drift = 0.1      # Small drift to ignore minor deviations, adjust as needed
    """
    pos_cusum = 0.0
    first_strong_peak = 0

    for i in range(1, len(signal_data)):
        pos_cusum = max(0, pos_cusum + signal_data[i] - drift)
        if first_strong_peak == 0 and pos_cusum > threshold_pos:
            first_strong_peak = i

    return first_strong_peak


def cusum(signal_data, threshold=0.4, drift=0.1):
    # Parameters for CUSUM
    # threshold = 0.4  # Adjust based on noise level and desired sensitivity
    # drift = 0.1     # Small drift to ignore minor deviations, adjust as needed

    # Initialize positive and negative CUSUM
    pos_cusum = np.zeros_like(signal_data)
    neg_cusum = np.zeros_like(signal_data)

    # Calculate CUSUM iteratively
    for i in range(1, len(signal_data)):
        pos_cusum[i] = max(0, pos_cusum[i - 1] + signal_data[i] - drift)
        neg_cusum[i] = min(0, neg_cusum[i - 1] + signal_data[i] + drift)
        # Check if CUSUM exceeds threshold for significant upward shift
        if pos_cusum[i] > threshold:
            first_strong_peak = i
            break

    # 1. Calc deviation from drift
    devaitions = np.maximum(0, signal_data - drift)
    # 2. Calc cum sum for positive CUSUM
    pos_cusum = np.maximum.accumulate(devaitions.cumsum())
    # 3. Find first index where CUSUM exceeds threshold
    first_strong_peak = np.argmax(pos_cusum > threshold)

    plt.figure()
    plt.plot(signal_data)
    plt.plot(pos_cusum)
    plt.scatter(
        [first_strong_peak], [signal_data[first_strong_peak]], c="r", marker="x"
    )

    return first_strong_peak


def plot_transform(rfs, **kwargs):
    n = len(rfs)
    f, ax = plt.subplots(n, 1, sharex=True, **kwargs)
    ax = ax.flatten()
    for i, rf in enumerate(rfs):
        env = rf[0]
        env[:200] = 0
        ax[i].plot(env, label="Env")
        # ax[i].plot(transform(env), label="Transformed")
        # ax[i].set(ylim=[0, 1])

        # first_peak = cusum(env)
        first_peak = cusum_fast(env)
        ax[i].scatter([first_peak], [env[first_peak]], c="r", marker="x")
        # ax[i].scatter([first_trough], [env[first_trough]], c="r", marker="x")

        # last_peak = cusum(env[::-1])
        # last_peak = len(env) - last_peak
        # ax[i].scatter([last_peak], [env[last_peak]], c="r", marker="x")

    ax[0].legend()


plot_transform(rfs, figsize=(6, 6))

# %%
cusum(rfs[0][0])
# %timeit cusum(rfs[0][0])
cusum_fast(rfs[0][0])


# %%
# %timeit cusum_fast(rfs[0][0])
# %%
def fix_surface_idx_missing(idx):
    """
    Fix cases of missing surface (group of 0s) or incorrect deeper surface
    (disjoint lines)
    """
    ### First clear disjoint surfaces
    MAX_DISTANCE = 500

    def _prev_disjoint(i_prev, i_curr, max_distance):
        return abs(idx[i_curr] - idx[i_prev]) > max_distance

    # # Assume idx[0] is correct...
    # i = 1
    # last_good_i = i - 1
    # while i < len(idx):
    #     # Skip over zeros
    #     while i < len(idx) and idx[i] == 0:
    #         i += 1

    #     # Follow next disjoint segment
    #     # print(f"{i=} {last_good_i=}")
    #     if i < len(idx):
    #         if _prev_disjoint(last_good_i, i, MAX_DISTANCE + i - last_good_i):
    #             disjoint_start = i
    #             i += 1
    #             while i < len(idx) and not _prev_disjoint(
    #                 i - 1, i, MAX_DISTANCE + i - last_good_i
    #             ):
    #                 i += 1
    #             disjoint_end = i
    #             print("Disjoint ", (disjoint_start, disjoint_end))
    #             idx[disjoint_start:disjoint_end] = 0
    #         else:
    #             i += 1
    #             last_good_i = i - 1

    def _interp(v1: float, v2: float, n: int):
        """
        Linearly interpolate n points inside the range [v1, v2],
        where n is the number of points in between.
        For example, _interp(1., 2., 3) == [1.25, 1.5, 1.75]
        """

        # x = np.arange(1, n + 1)
        # xp = [0, n + 1]
        # fp = [v1, v2]
        # res = np.interp(x, xp, fp)

        res = np.linspace(v1, v2, n + 2)
        res = res[1:-1]
        return res

    def _interp_next_gap_inplace(idx, i):
        """"""
        while i < len(idx):
            if idx[i] == 0:
                # Look ahead until we find first nonzero
                i_start = i
                while i < len(idx) and idx[i] == 0:
                    i += 1
                i_end = i

                # 2
                if i_start > 0 and i_end < idx.size:
                    n = i_end - i_start
                    v1 = idx[i_start - 1]
                    v2 = idx[i_end]

                    res = _interp(v1, v2, n)
                    idx[i_start:i_end] = res
            else:
                i += 1
        return idx, i

    # Cases
    # 1. i_start == 0 (first zero)
    #       Move on and wait til the end. Then wrap i_start to the end
    # 2. i_start != 0, i_end != end (zeros in middle)
    #       Interpolate
    # 3. i_start != 0, i_end = end (last zero)
    #       Interpolate to start

    # Case 1. Ignore for now and do Case 2 first
    i = 0
    while i < len(idx) and idx[i] == 0:
        i += 1

    # Case 2
    while i < len(idx):
        idx, i = _interp_next_gap_inplace(idx, i)

    # Do Case 1 and Case 3
    if idx[0] == 0 or idx[-1] == 0:
        # Find last nonzero
        i = len(idx) - 1
        while idx[i] == 0 and i >= 0:
            i -= 1
        if i < 0:
            # All indices < 0. No surface found.
            # TODO. should handle this error explicitly
            return idx

        n_pts_rotate = len(idx) - i
        idx = np.roll(idx, n_pts_rotate)
        _interp_next_gap_inplace(idx, 1)
        idx = np.roll(idx, -n_pts_rotate)

    return idx


# %%
def make_radial_v2(img, padding: int = 0):
    """
    Input image has alines for rows
    """
    if padding != 0:
        img = cv2.copyMakeBorder(img, 0, 0, padding, 0, 0)

    h, w = img.shape
    r = min(h, w)
    center = (r / 2, r / 2)
    dsize = (r, r)
    maxRadius = r / 2
    img = cv2.warpPolar(
        img, dsize, center, maxRadius, cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS
    )
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# %%
def find_surface(rf):
    surface_idx = np.zeros(len(rf))
    for i in range(len(rf)):
        rf[i, :200] = 0
        surface_idx[i] = cusum_fast(rf[i], 0.5, 0.005)
    surface_idx = fix_surface_idx_missing(surface_idx)
    return surface_idx


rf = rfs[3]
rf[:, :200] = 0

rf_log, _ = log_compress(rf, 0.002, 35)

img = rf_log
img = cv2.resize(img, (1000, 1000))

surface_idx = find_surface(rf)

# surface = signal.medfilt(surface_idx, 7)
surface = signal.savgol_filter(surface_idx, 99, 2, mode="wrap")
surface_1000 = surface / rf_log.shape[1] * 1000
# plt.figure()
# plt.plot(surface)


plt.figure()
plt.imshow(img.T, "gray")
plt.plot(surface_1000)


# %%
def warp(x, y, center, dsize, maxRadius):
    Ix = x - center[0]
    Iy = y - center[1]

    angle = np.arctan2(Iy, Ix)
    magnitude = np.sqrt(Ix**2 + Iy**2)

    # dsize.height / 2pi
    Kangle = dsize[1] / (2 * np.pi)
    # dsize.width / maxRadius
    Klin = dsize[0] / maxRadius

    phi = Kangle * angle
    rho = Klin * magnitude
    return rho, phi


def warp_inverse(rho, phi, center, dsize, maxRadius):
    # dsize.height / 2pi
    Kangle = dsize[1] / (2 * np.pi)
    # dsize.width / maxRadius
    Klin = dsize[0] / maxRadius

    angle = phi / Kangle
    magnitude = rho / Klin

    x = np.round(center[0] + magnitude * np.cos(angle))
    y = np.round(center[1] + magnitude * np.sin(angle))
    return x, y


x, y = warp_inverse(surface_1000, np.arange(1000), (500, 500), (1000, 1000), 500)


plt.imshow(make_radial_v2(img), "gray")
plt.plot(y, 1000 - x)

# %%
img_flat = np.zeros_like(img)

for i in range(len(rf_log)):
    img_flat[i] = np.roll(img[i], 500 - round(surface_1000[i]))
    # rf_log_flat[i] = rf_log[i]

plt.figure()
plt.imshow(img_flat, "gray")

# %%
surface_flat = np.ones(1000) * 500
x, y = warp_inverse(surface_flat, np.arange(1000), (500, 500), (1000, 1000), 500)

plt.imshow(make_radial_v2(img_flat), "gray")
plt.plot(y, 1000 - x)

# %%
