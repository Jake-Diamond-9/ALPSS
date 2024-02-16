import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.signal import hilbert
import matplotlib.pyplot as plt


# general function for for a sinusoid
def sin_func(x, a, b, c, d):
    return a * np.sin(2 * np.pi * b * x + c) + d


# get the indices for the upper and lower envelope of the voltage signal
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    # https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal

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
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global max of dmax-chunks of locals max
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax


# function to estimate the instantaneous uncertainty for all points in time
def instantaneous_uncertainty_analysis(sdf_out, vc_out, cen, **inputs):
    # unpack needed variables
    lam = inputs['lam']
    smoothing_window = inputs['smoothing_window']
    fs = sdf_out['fs']
    time = sdf_out['time']
    # voltage = sdf_out['voltage']
    voltage_filt = vc_out['voltage_filt']
    t_doi_start = sdf_out['t_doi_start']
    t_doi_end = sdf_out['t_doi_end']
    carrier_band_time = inputs['carrier_band_time']
    # t_start_corrected = sdf_out['t_start_corrected']
    # t_before = inputs['t_before']
    # t_after = inputs['t_after']

    # take only real component of the filtered voltage signal
    voltage_filt = np.real(voltage_filt)

    # amount of time from the beginning of the voltage signal to analyze for noise
    t_take = carrier_band_time
    steps_take = int(t_take * fs)

    # get the data for only the beginning section of the signal
    time_cut = time[0:steps_take]
    # voltage_cut = voltage[0:steps_take]
    voltage_filt_cut = voltage_filt[0:steps_take]

    # fit a sinusoid to the data
    popt, pcov = curve_fit(sin_func, time_cut, voltage_filt_cut, p0=[0.1, cen, 0, 0])

    # calculate the fitted curve
    volt_fit = sin_func(time_cut, popt[0], popt[1], popt[2], popt[3])

    # calculate the residuals
    noise = voltage_filt_cut - volt_fit

    # calculate the envelope indices of the originally imported voltage data (and now filtered) using the stack
    # overflow code
    lmin, lmax = hl_envelopes_idx(voltage_filt, dmin=1, dmax=1, split=False)

    # interpolate the voltage envelope to every time point
    env_max_interp = np.interp(time, time[lmax], voltage_filt[lmax])
    env_min_interp = np.interp(time, time[lmin], voltage_filt[lmin])

    # calculate the estimated amplitude at every time
    inst_amp = env_max_interp - env_min_interp

    # calculate the estimated noise fraction at every time
    # https://doi.org/10.1063/12.0000870
    inst_noise = np.std(noise) / (inst_amp / 2)

    # calculate the frequency and velocity uncertainty
    tau = smoothing_window / fs
    freq_uncert_scaling = (1 / np.pi) * (np.sqrt(6 / (fs * (tau ** 3))))
    freq_uncert = inst_noise * freq_uncert_scaling
    vel_uncert = freq_uncert * (lam / 2)

    # find max noise on the domain of interest
    t_doi_start_idx = np.argmin(np.abs(time - t_doi_start))
    t_doi_end_idx = np.argmin(np.abs(time - t_doi_end))
    doi_max_noise = np.max(inst_noise[t_doi_start_idx:t_doi_end_idx])

    # print(inst_noise)
    # print(tau)
    # print(freq_uncert)
    # print(vel_uncert)

    '''
    # plotting
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, dpi=300)
    ax1.plot(time_cut / 1e-9, voltage_filt_cut * 1e3, c='tab:blue')
    ax1.plot(time_cut / 1e-9, volt_fit * 1e3, c='tab:orange')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Voltage (mV)')

    ax2.hist(noise * 1e3, bins=30, rwidth=0.8)
    ax2.set_xlabel('Noise (V)')
    ax2.set_ylabel('Counts')

    ax3.plot(time / 1e-9, voltage_filt * 1e3, c='tab:blue')
    ax3.plot(time / 1e-9, env_max_interp * 1e3, c='tab:red')
    ax3.plot(time / 1e-9, env_min_interp * 1e3, c='tab:red')
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Voltage (mV)')
    ax3.set_xlim([t_doi_start / 1e-9, t_doi_end / 1e-9])

    ax4.plot(time / 1e-9, inst_noise * 100)
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Noise Fraction (%)')
    ax4.set_xlim([t_doi_start / 1e-9, t_doi_end / 1e-9])
    ax4.set_ylim([0, doi_max_noise * 110])
    # ax4.set_ylim([0, 100])

    ax5.plot(time / 1e-9, freq_uncert / 1e9)
    ax5.set_xlabel('Time (ns)')
    ax5.set_ylabel('Frequency Uncert (GHz)')
    ax5.set_xlim([t_doi_start / 1e-9, t_doi_end / 1e-9])
    ax5.set_ylim([0, (doi_max_noise * freq_uncert_scaling * 1.1) / 1e9])

    ax6.plot(time / 1e-9, vel_uncert)
    ax6.set_xlabel('Time (ns)')
    ax6.set_ylabel('Velocity Uncert (m/s)')
    ax6.set_xlim([t_doi_start / 1e-9, t_doi_end / 1e-9])
    ax6.set_ylim([0, (doi_max_noise * freq_uncert_scaling * (lam / 2) * 1.1)])

    plt.tight_layout()
    plt.show()
    '''

    # dictionary to return outputs
    iua_out = {
        'time_cut': time_cut,
        'popt': popt,
        'pcov': pcov,
        'volt_fit': volt_fit,
        'noise': noise,
        'env_max_interp': env_max_interp,
        'env_min_interp': env_min_interp,
        'inst_amp': inst_amp,
        'inst_noise': inst_noise,
        'doi_max_noise': doi_max_noise,
        'freq_uncert_scaling': freq_uncert_scaling,
        'freq_uncert': freq_uncert,
        'vel_uncert': vel_uncert
    }

    return iua_out
