# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# FIR filter
def apply_fir_filt(x, kernel):
    x_filt = np.empty_like(x, dtype=np.double)
    for i in range(x.shape[0]):
        x_filt[i] = np.convolve(x[i], kernel, "same")
    return x_filt


def recon(rf, kernel):
    rf_filt = apply_fir_filt(rf, kernel)
    rf_env = np.abs(signal.hilbert(rf_filt))
    return rf_env


def log_compress(x, db):
    x = x / x.max()
    x_log = 255 * (20 / db * np.log10(x) + 1)
    x_log = np.where(x_log < 1, 0, x_log)
    return x_log


# %%
root = Path.home() / "Downloads/135245/135245"
path = root / "NormalUS4.bin"
rf = np.fromfile(path, dtype=np.dtype(">d"), sep="")
rf = rf.reshape((1000, 2 * 3200))

# %%
plt.plot(rf[0])

# %%
kernel = signal.firwin2(65, freq=[0, 0.1, 0.3, 1], gain=[0, 1, 1, 0])
rf_filt = apply_fir_filt(rf, kernel)


# %%
rf_env = recon(rf, kernel)
rf_log = log_compress(rf_env, 25)
plt.imshow(rf_log.T)

# %%
def firwin2(
    numtaps,
    freq,
    gain,
    nfreqs=None,
    fs=2.0,
):
    """
    FIR filter design using the window method.

    From the given frequencies `freq` and corresponding gains `gain`,
    this function constructs an FIR filter with linear phase and
    (approximately) the given frequency response.

    Parameters
    ----------
    numtaps : int
        The number of taps in the FIR filter.  `numtaps` must be less than
        `nfreqs`.
    freq : array_like, 1-D
        The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
        Nyquist.  The Nyquist frequency is half `fs`.
        The values in `freq` must be nondecreasing. A value can be repeated
        once to implement a discontinuity. The first value in `freq` must
        be 0, and the last value must be ``fs/2``. Values 0 and ``fs/2`` must
        not be repeated.
    gain : array_like
        The filter gains at the frequency sampling points. Certain
        constraints to gain values, depending on the filter type, are applied,
        see Notes for details.
    nfreqs : int, optional
        The size of the interpolation mesh used to construct the filter.
        For most efficient behavior, this should be a power of 2 plus 1
        (e.g, 129, 257, etc). The default is one more than the smallest
        power of 2 that is not less than `numtaps`. `nfreqs` must be greater
        than `numtaps`.
    window : string or (string, float) or float, or None, optional
        Window function to use. Default is "hamming". See
        `scipy.signal.get_window` for the complete list of possible values.
        If None, no window function is applied.
    nyq : float, optional
        *Deprecated. Use `fs` instead.* This is the Nyquist frequency.
        Each frequency in `freq` must be between 0 and `nyq`.  Default is 1.
    antisymmetric : bool, optional
        Whether resulting impulse response is symmetric/antisymmetric.
        See Notes for more details.
    fs : float, optional
        The sampling frequency of the signal. Each frequency in `cutoff`
        must be between 0 and ``fs/2``. Default is 2.

    Returns
    -------
    taps : ndarray
        The filter coefficients of the FIR filter, as a 1-D array of length
        `numtaps`.

    See also
    --------
    firls
    firwin
    minimum_phase
    remez

    Notes
    -----
    From the given set of frequencies and gains, the desired response is
    constructed in the frequency domain. The inverse FFT is applied to the
    desired response to create the associated convolution kernel, and the
    first `numtaps` coefficients of this kernel, scaled by `window`, are
    returned.

    The FIR filter will have linear phase. The type of filter is determined by
    the value of 'numtaps` and `antisymmetric` flag.
    There are four possible combinations:

       - odd  `numtaps`, `antisymmetric` is False, type I filter is produced
       - even `numtaps`, `antisymmetric` is False, type II filter is produced
       - odd  `numtaps`, `antisymmetric` is True, type III filter is produced
       - even `numtaps`, `antisymmetric` is True, type IV filter is produced

    Magnitude response of all but type I filters are subjects to following
    constraints:

       - type II  -- zero at the Nyquist frequency
       - type III -- zero at zero and Nyquist frequencies
       - type IV  -- zero at zero frequency

    .. versionadded:: 0.9.0

    References
    ----------
    .. [1] Oppenheim, A. V. and Schafer, R. W., "Discrete-Time Signal
       Processing", Prentice-Hall, Englewood Cliffs, New Jersey (1989).
       (See, for example, Section 7.4.)

    .. [2] Smith, Steven W., "The Scientist and Engineer's Guide to Digital
       Signal Processing", Ch. 17. http://www.dspguide.com/ch17/1.htm

    Examples
    --------
    A lowpass FIR filter with a response that is 1 on [0.0, 0.5], and
    that decreases linearly on [0.5, 1.0] from 1 to 0:

    >>> from scipy import signal
    >>> taps = signal.firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    >>> print(taps[72:78])
    [-0.02286961 -0.06362756  0.57310236  0.57310236 -0.06362756 -0.02286961]

    """
    nyq = 0.5 * fs

    if len(freq) != len(gain):
        raise ValueError("freq and gain must be of same length.")

    if nfreqs is None:
        nfreqs = 1 + 2 ** int(np.ceil(np.log2(numtaps)))

    # Linearly interpolate the desired response on a uniform mesh `x`.
    x = np.linspace(0.0, nyq, nfreqs)
    print(f"{x=}")
    fx = np.interp(x, freq, gain)
    print(f"{fx=}")

    # Adjust the phases of the coefficients so that the first `ntaps` of the
    # inverse FFT are the desired filter coefficients.
    shift = np.exp(-(numtaps - 1) / 2.0 * 1.0j * np.pi * x / nyq)
    fx2 = fx * shift

    # Use irfft to compute the inverse FFT.
    out_full = np.fft.irfft(fx2)

    wind = signal.windows.get_window("hamming", numtaps, fftbins=False)

    # Keep only the first `numtaps` coefficients in `out`, and multiply by
    # the window.
    out = out_full[:numtaps] * wind
    # plt.plot(out)

    return out


# kernel = signal.firwin2(65, freq=[0, 0.1, 0.3, 1], gain=[0, 1, 1, 0])
# kernel_ = firwin2(65, freq=[0, 0.1, 0.3, 1], gain=[0, 1, 1, 0])
# np.allclose(kernel, kernel_)

np.allclose(firwin2(15, freq=[0, 0.1, 0.3, 1], gain=[0, 1, 1, 0]), signal.firwin2(15, freq=[0, 0.1, 0.3, 1], gain=[0, 1, 1, 0]))

# %%
inp = np.sin(np.linspace(0, 2 * np.pi, 50))
kernel = firwin2(15, [0, 0.1, 0.3, 1.0], [0, 1, 1, 0])
out = np.convolve(inp, kernel, "same")

plt.plot(inp, label="inp")
plt.plot(out)

# %%
out

# %%

# %%
# Envelope detection
rf_env = np.abs(signal.hilbert(rf_filt))

plt.plot(rf_env[0])

# %%
env = np.fromfile("../build/clang-debug/libarpam/output.bin", np.float64)

env = env.reshape(1000, 6400)

# %%
i = 100
np.allclose(env[i, :10], rf_env[i, :10])

# %%
np.allclose(env, rf_env)

# %%
i = 1
plt.plot(env[i, :100])
plt.plot(rf_env[i, :100])

# %%
plt.imshow(np.log(env))

# %%


# %%
_rf = np.fromfile("../build/clang-debug/libarpam/rf.bin", np.float64)
plt.plot(_rf)

# %%
_rf_filt_padded = np.fromfile(
    "../build/clang-debug/libarpam/rf_filt_padded.bin", np.float64
)
plt.plot(_rf_filt_padded)

# %%
_rf_filt = np.fromfile("../build/clang-debug/libarpam/rf_filt.bin", np.float64)
plt.plot(_rf_filt)

# %%
output = np.fromfile("../build/clang-debug/arpam_gui_glfw_vulkan/output.bin", np.uint8)
output = output.reshape(1000, 6400)

# %%
# plt.imshow(output)
plt.imshow(output, "gray", extent=[-1, 1, -1, 1])


# %%
plt.imshow(output)


# %%
# Log compression
rf_log = log_compress(rf_env, 45)
plt.plot(rf_log[0])

# %%

# %%
plt.imshow(log_compress(env, 45))

# %%
plt.imshow(rf_log.T, "gray", extent=[-1, 1, -1, 1])

# %%
rf_env[0][:10]

# %%

# %%

# %%
## New Ex Vivo
path = Path("/Volumes/taylor/20240319_exvivo/")
bins = list(path.glob("*"))

raw = np.memmap(bins[1], dtype="uint16")
raw.size

# %%
i = 8192 * 1000 * 10
plt.plot(raw[i:i + 8192])

# %%
npoints_PA = 2730
npoints_US = 5460 + 2
npoints_PAUS = 8192

# %%
n_bscans = raw.size // (npoints_PAUS * 1000)
print(n_bscans)
all_rf = raw.reshape((n_bscans, 1000, npoints_PAUS))

# %%
plt.figure()
plt.plot(all_rf[100, 0,])
plt.xlabel("points")

# %%
rf = all_rf[100, 0]
rf = (rf / 2**15) - 1
plt.plot(rf)

# %%
rf.max()

# %%
kernel_PA = signal.firwin2(65, [0, 0.03, 0.035, 0.2, 0.22, 1], [0, 0, 1, 1, 0, 0])
kernel_US = signal.firwin2(65, [0, 0.1, 0.3, 1], [0, 1, 1, 0])

# %%
scan_i = 100

rf_PA = all_rf[scan_i, :, :npoints_PA]
rf_US = all_rf[scan_i, :, npoints_PA:]

PA_env = recon(rf_PA, kernel_PA)
US_env = recon(rf_US, kernel_US)

PA_log = log_compress(PA_env, 25)
US_log = log_compress(US_env, 30)

fig, ax = plt.subplots(1, 2, figsize=(6, 6))
ax[0].imshow(PA_log.T, "gray")
ax[1].imshow(US_log.T, "gray")

# %%
%matplotlib inline
plt.figure()
plt.plot(US_env[0])

# %%
# PA_env.dtype
rf_PA.dtype

# %%
plt.plot(rf_PA[0])
