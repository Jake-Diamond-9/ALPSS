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
    fs = sdf_out['fs']
    time = sdf_out['time']
    voltage = sdf_out['voltage']
    voltage_filt = vc_out['voltage_filt']

    # take only real component of the filtered voltage signal
    voltage_filt = np.real(voltage_filt)

    # amount of time from the beginning of the voltage signal to analyze for noise
    t_take = 250e-9
    steps_take = int(t_take*fs)

    # get the data for only the beginning section of the signal
    time_cut = time[0:steps_take]
    voltage_cut = voltage[0:steps_take]
    voltage_filt_cut = voltage_filt[0:steps_take]


    # fit a sinusoid to the data
    popt, pcov = curve_fit(sin_func, time_cut, voltage_cut, p0=[0.1, cen, 0, 0])
    popt_filt, pcov_filt = curve_fit(sin_func, time_cut, voltage_filt_cut, p0=[0.1, cen, 0, 0])


    # calculate the fitted curve
    volt_fit = sin_func(time_cut, popt[0], popt[1], popt[2], popt[3])
    volt_fit_filt = sin_func(time_cut, popt_filt[0], popt_filt[1], popt_filt[2], popt_filt[3])

    # calculate the residuals
    noise = voltage_cut - volt_fit
    noise_filt = voltage_filt_cut - volt_fit_filt

    '''
    # calculate the envelope indices of the originally imported voltage data (and now filtered) using the stack
    # overflow code
    lmin, lmax = hl_envelopes_idx(voltage_cut, dmin=10, dmax=10, split=False)

    # interpolate the voltage envelope to every time point
    env_max_interp = np.interp(time, time[lmax], voltage_filt_abs[lmax])
    env_min_interp = np.interp(time, time[lmin], voltage_filt_abs[lmin])
    '''


    # plotting
    y_lim_volt = [-300, 300]
    x_lim_noise = [-50, 50]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, dpi=300)
    ax1.plot(time_cut / 1e-9, voltage_cut*1e3, c='tab:blue')
    ax1.plot(time_cut / 1e-9, volt_fit*1e3, c='tab:orange')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_ylim(y_lim_volt)


    ax2.plot(time_cut / 1e-9, voltage_filt_cut * 1e3, c='tab:blue')
    ax2.plot(time_cut / 1e-9, volt_fit_filt*1e3, c='tab:orange')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (mV)')
    ax2.set_ylim(y_lim_volt)


    ax3.hist(noise*1e3, bins=30, rwidth=0.8)
    ax3.set_xlabel('Noise (V)')
    ax3.set_ylabel('Counts')
    ax3.set_xlim(x_lim_noise)

    ax4.hist(noise_filt * 1e3, bins=30, rwidth=0.8)
    ax4.set_xlabel('Noise (V)')
    ax4.set_ylabel('Counts')
    ax4.set_xlim(x_lim_noise)


    plt.tight_layout()
    plt.show()

