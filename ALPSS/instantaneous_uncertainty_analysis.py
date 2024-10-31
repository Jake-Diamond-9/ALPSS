import numpy as np
from scipy.optimize import curve_fit
import traceback

# gaussian distribution
def gauss(x, amp, sigma, mu):
    f = (amp / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return f

# general function for a sinusoid
def sin_func(x, a, b, c, d):
    return a * np.sin(2 * np.pi * b * x + c) + d

# calculate the fwhm of a gaussian distribution
def fwhm(
    smoothing_window, smoothing_wid, smoothing_amp, smoothing_sigma, smoothing_mu, fs
):
    # x points for the gaussian weights
    x = np.linspace(-smoothing_wid, smoothing_wid, smoothing_window)

    # calculate the gaussian weights
    weights = gauss(x, smoothing_amp, smoothing_sigma, smoothing_mu)

    # calculate the half max
    half_max = ((np.max(weights) - np.min(weights)) / 2) + np.min(weights)

    # calculate the fwhm of the gaussian weights for the normalized x points
    fwhm_norm = 2 * np.abs(x[np.argmin(np.abs(weights - half_max))])

    # scale the fwhm to the number of points being used for the smoothing window
    fwhm_pts = (fwhm_norm / (smoothing_wid * 2)) * smoothing_window

    # calculate the time span of the fwhm of the gaussian weights
    fwhm = fwhm_pts / fs

    return fwhm

# get the indices for the upper and lower envelope of the voltage signal
# https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[
        [i + np.argmin(s[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]
    ]
    # global max of dmax-chunks of locals max
    lmax = lmax[
        [i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
    ]

    return lmin, lmax

# function to estimate the instantaneous uncertainty for all points in time
def instantaneous_uncertainty_analysis(sdf_out, vc_out, cen, **inputs):
    # unpack needed variables
    lam = inputs["lam"]
    smoothing_window = inputs["smoothing_window"]
    smoothing_wid = inputs["smoothing_wid"]
    smoothing_amp = inputs["smoothing_amp"]
    smoothing_sigma = inputs["smoothing_sigma"]
    smoothing_mu = inputs["smoothing_mu"]
    fs = sdf_out["fs"]
    time = sdf_out["time"]
    time_f = vc_out["time_f"]
    voltage_filt = vc_out["voltage_filt"]
    time_start_idx = vc_out["time_start_idx"]
    time_end_idx = vc_out["time_end_idx"]
    carrier_band_time = inputs["carrier_band_time"]

    # take only real component of the filtered voltage signal
    voltage_filt = np.real(voltage_filt)

    # amount of time from the beginning of the voltage signal to analyze for noise
    t_take = carrier_band_time
    steps_take = int(t_take * fs)

    # get the data for only the beginning section of the signal
    time_cut = time[0:steps_take]
    voltage_filt_early = voltage_filt[0:steps_take]

    try:
        # fit a sinusoid to the data
        popt, pcov = curve_fit(
            sin_func, time_cut, voltage_filt_early, p0=[0.1, cen, 0, 0]
        )
    except Exception:
        # if sin fitting doesn't work set the fitting parameters to be zeros
        print(traceback.format_exc())
        popt = [0, 0, 0, 0]
        pcov = [0, 0, 0, 0]

    # calculate the fitted curve
    volt_fit = sin_func(time_cut, popt[0], popt[1], popt[2], popt[3])

    # calculate the residuals
    noise = voltage_filt_early - volt_fit

    # get data for only the doi of the voltage
    voltage_filt_doi = voltage_filt[time_start_idx:time_end_idx]

    # calculate the envelope indices of the originally imported voltage data (and now filtered) using the stack
    # overflow code
    lmin, lmax = hl_envelopes_idx(voltage_filt_doi, dmin=1, dmax=1, split=False)

    # interpolate the voltage envelope to every time point
    env_max_interp = np.interp(time_f, time_f[lmax], voltage_filt_doi[lmax])
    env_min_interp = np.interp(time_f, time_f[lmin], voltage_filt_doi[lmin])

    # calculate the estimated peak to peak amplitude at every time
    inst_amp = env_max_interp - env_min_interp

    # calculate the estimated noise fraction at every time
    # https://doi.org/10.1063/12.0000870
    inst_noise = np.std(noise) / (inst_amp / 2)

    # calculate the frequency and velocity uncertainty
    # https://doi.org/10.1063/12.0000870
    # take the characteristic time to be the fwhm of the gaussian weights used for smoothing the velocity signal
    tau = fwhm(
        smoothing_window,
        smoothing_wid,
        smoothing_amp,
        smoothing_sigma,
        smoothing_mu,
        fs,
    )
    freq_uncert_scaling = (1 / np.pi) * (np.sqrt(6 / (fs * (tau**3))))
    freq_uncert = inst_noise * freq_uncert_scaling
    vel_uncert = freq_uncert * (lam / 2)

    # dictionary to return outputs
    iua_out = {
        "time_cut": time_cut,
        "popt": popt,
        "pcov": pcov,
        "volt_fit": volt_fit,
        "noise": noise,
        "env_max_interp": env_max_interp,
        "env_min_interp": env_min_interp,
        "inst_amp": inst_amp,
        "inst_noise": inst_noise,
        "tau": tau,
        "freq_uncert_scaling": freq_uncert_scaling,
        "freq_uncert": freq_uncert,
        "vel_uncert": vel_uncert,
    }

    return iua_out