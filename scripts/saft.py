# %%
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class SaftDelayParams:
    """
    SAFT parameters relating to transducer geometry, rotation geometry, and illumination geometry
    """

    rt: float
    "[mm] distance from axis of rotation to transducer surface"
    vs: float
    "[m/s] sound speed"
    dt: float
    "[s] timestep"

    da: float
    "[rad] angle step size in each rotation"

    f: float
    "[mm] transducer focal length"
    d: float
    "[mm] transducer diameter"
    angle: float
    "[rad] transducer focus angle"

    angle_light: float
    "[rad] illumination angle"

    @property
    def dr(self):
        "[mm] spatial step size"
        return self.vs * self.dt * 1e3

    @classmethod
    def make(cls) -> SaftDelayParams:
        return cls(
            rt=6.2,
            vs=1.5e3,
            dt=1.0 / 180e6,
            da=2 * np.pi / 1000,
            f=15.0,
            d=8.5,
            angle=np.arcsin(8.5 / (2 * 15.0)),
            angle_light=np.deg2rad(5),
        )


def compute_SAFT_time_delay(p: SaftDelayParams, z_start: int = -1, z_end: int = -1):
    # [pts] z start and end points of SAFT.
    # By default start = (half focal distance), end = (1.5x focal distance)
    if z_start < 0:
        z_start = round((p.f * 0.25) / p.dr)
    if z_end < 0:
        z_end = round((p.f * 1.4) / p.dr)

    max_saft_lines = 12

    # number of saft lines as a func of z
    n_saft_lines = np.zeros(z_end - z_start, dtype=np.uint8)
    time_delay = np.zeros((z_end - z_start, max_saft_lines), order="F")

    for j in range(1, max_saft_lines):

        for i in range(z_start, z_end):  # depth
            dr1 = i * p.dr
            r = p.rt + i * p.dr

            # relative position to the transducer center
            # dr2 and ang2
            ang1 = j * p.da
            dr2 = math.sqrt(r**2 + p.rt**2 - 2 * r * p.rt * math.cos(ang1))
            ang2 = math.pi - math.acos((p.rt**2 + dr2**2 - r**2) / (2 * p.rt * dr2))

            # Determine if point is within the light beam field
            if ang2 >= p.angle_light:
                continue

            # Determine if point is within the transducer field

            # distance to focus
            dr3 = math.sqrt(p.f**2 + dr2**2 - 2 * p.f * dr2 * math.cos(ang2))

            # angle wrt focal line
            ang3 = math.acos((p.f**2 + dr3**2 - dr2**2) / (2 * p.f * dr3))

            if dr3 <= p.f and ang3 <= p.angle:
                time_delay[i - z_start, j] = (abs(p.f - dr1) - dr3) / p.dr
                n_saft_lines[i - z_start] += 1
            elif math.pi - ang3 <= p.angle:
                time_delay[i - z_start, j] = (dr3 - abs(p.f - dr1)) / p.dr
                n_saft_lines[i - z_start] += 1

    return time_delay, n_saft_lines, (z_start, z_end)


p = SaftDelayParams.make()

zpair = [769, 2450]
time_delay, n_saft_lines, zpair = compute_SAFT_time_delay(p, *zpair)


# %%
import numba

@numba.jit()
def apply_saft(
    PA: np.ndarray,
    time_delay: np.ndarray,
    n_saft_lines: np.ndarray,
    z_start: int,
    z_end: int,
):
    "PA is column major"
    (N_ascan, n_pts) = PA.shape
    PA_saft = PA.copy()

    # normalization factor (w.r.t num of lines summed)
    NF_saft = np.ones_like(PA)

    CF_denom = PA**2

    for i_ascan in range(N_ascan):
        for iz in range(z_start, z_end):
            for di_saft in range(n_saft_lines[iz - z_start]):
                iz_delayed = round(iz + time_delay[iz - z_start, di_saft])
                if iz_delayed >= n_pts:
                    continue
                PA_val = PA[i_ascan, iz_delayed]

                # i_saft = i_ascan - di_saft % N_ascan
                i_saft = (i_ascan - di_saft) % N_ascan

                PA_saft[i_saft, iz] += PA_val
                CF_denom[i_saft, iz] += PA_val ** 2
                NF_saft[i_saft, iz] += 1

                # i_saft = i_ascan + di_saft % N_ascan
                i_saft = (i_ascan + di_saft) % N_ascan
                PA_saft[i_saft, iz] += PA_val
                CF_denom[i_saft, iz] += PA_val ** 2
                NF_saft[i_saft, iz] += 1

    ## TODO
    # 2. apply normal delay and sum in focal region

    # need to guard against div by zero here
    # NF_saft could be zero
    CF = PA_saft**2 / (CF_denom * NF_saft)
    for i in range(CF.shape[0]):
        for j in range(CF.shape[1]):
            if np.isnan(CF[i, j]):
                CF[i, j] = 1

    PA_saft_cf = PA_saft * CF / NF_saft

    # TODO magnitude correction by depth
    return PA_saft, PA_saft_cf, NF_saft, CF



# %%
rf = np.fromfile("/Volumes/TDrive/Taylor_SAFT/Taylor_SAFT/013941PA.bin", dtype="double")
rf = np.fromfile("/Volumes/TDrive/Taylor_SAFT/Taylor_SAFT/013941PA.bin", dtype="double")
rf = rf.reshape((2730, 1000, rf.size // 1000 // 2730), order="F")
rf = rf[:, :, 0].T
rf = rf[:, :2500]

# rf = ((rf + 1) * 2**15).astype(np.uint16)
# %%
pa_saft, _pa_saft_cf, NF_saft, CF = apply_saft(rf, time_delay, n_saft_lines, zpair[0], zpair[1])

# np.allclose(_pa_saft_cf, pa_saft_cf)

# %%
# plt.imshow(NF_saft)
plt.imshow(pa_saft)

# %%
i = 224
plt.figure()
# plt.plot(rf[i, :], label="rf")
plt.plot(pa_saft[i, :], label="rf_saft")
plt.plot(pa_saft_cf[i, :], label="rf_saft_cf")

plt.axvline(zpair[0], c="k")
plt.axvline(zpair[1], c="k")

plt.legend()


# %%
from pathlib import Path

p = (Path.home() / "code/cpp/ArpamGui/build/clang/ArpamGuiQt/RelWithDebInfo/ArpamGuiQt.app/Contents/MacOS/timeDelay.bin")
p.exists()

# %%
time_delay.shape

# %%
timeDelay = np.fromfile(p, np.float32)
timeDelay = timeDelay.reshape((15, timeDelay.size // 15))
timeDelay.shape

# %%
i = 15

plt.subplot(211)
plt.plot(timeDelay[i])

plt.subplot(212)
plt.plot(time_delay[:, i])

# %%
rf_saft_cf = np.fromfile("/src/ArpamGui/build/win64/libuspam/Release/rf_saft_cf.bin", np.float64)
rf_saft_cf = rf_saft_cf.reshape((2500, 1000), order="f")
rf_saft_cf.shape

# %%
rf_saft_cf.T - _pa_saft_cf

# %%
rf_saft_cf.max()

# %%
from scipy import signal
f, ax = plt.subplots(2, 1, sharex=True)
# for i in range(221, 225):
for i in range(18, 22):
    x = rf[i,:]
    ax[0].plot(x, label=f"{i}")
    x = np.abs(signal.hilbert(x))
    ax[1].plot(x)
f.legend()

# %%
import scipy.io as sio
af = sio.loadmat("/Volumes/TDrive/Taylor_SAFT/Taylor_SAFT/af.mat")["af"]
af = af.astype(np.float32)
plt.plot(af)
af.dtype

# %%
af.tofile("af.bin")

# %%
plt.imshow(pa_saft)
plt.colorbar()

# %%
from pathlib import Path
fname =Path.home() / "Downloads/single scatterer/112221PAUS.bin"
assert fname.exists()
scans = ulibarpam.load_scans(fname, 200)

scans = scans.astype(np.double)
background = scans.mean(axis=0)
background = signal.savgol_filter(background, 95, 2)
scans -= background

ioparams = ulibarpam.IOParams.default()
PA, US = ulibarpam.split_rf_USPA(scans, ioparams)

# %%
plt.figure()
# plt.subplot(211)
# plt.plot(PA[404])
# plt.subplot(212)
# plt.plot(PA.mean(axis=0))
plt.imshow(US)

# %%
plt.figure()
plt.imshow(PA)

# %%
f, ax = plt.subplots(2, 1, sharex=True)
# for i in range(221, 225):
for i in range(404, 409):
    x = PA[i,:]
    ax[0].plot(x, label=f"{i}")
    x = np.abs(signal.hilbert(x))
    ax[1].plot(x)

#  %%

# %%
plt.imshow(rf.T, "gray")
plt.gca().set_aspect(rf.shape[0] / rf.shape[1])
plt.colorbar()
plt.title("RF")

# %%
%matplotlib
import ulibarpam

params = ulibarpam.ReconParams.default()
k = ulibarpam.signal.firwin2(95, *params.filter_PA)

def recon(rf):
    img = ulibarpam.recon(rf - rf.mean(axis=0), k)
    img, db = ulibarpam.log_compress(img, 0.01, 45)
    return img, db

img, db = recon(rf_saft_cf)
print(db)
plt.figure()
plt.imshow(img.T, "gray")
plt.gca().set_aspect(img.shape[0] / img.shape[1])
plt.colorbar()

# %%
time_delay, n_saft_lines, zpair = compute_SAFT_time_delay(p)

# %%
# %matplotlib
plt.imshow(time_delay.T, interpolation="none")
plt.plot(n_saft_lines)
plt.gca().set_aspect(time_delay.shape[0] / time_delay.shape[1])
plt.colorbar()

# %%
rf_saft = apply_saft(rf, time_delay, n_saft_lines, zpair[0], zpair[1])

# %%
img, db = recon(rf_saft)
plt.imshow(img.T, "gray")
plt.gca().set_aspect(img.shape[0] / img.shape[1])
plt.colorbar()
plt.title("PA img")

# %%
np.allclose(n_saft_lines.astype(np.int64), n_saft_lines)

# %%
time_delay_old = time_delay.copy()

# %%
