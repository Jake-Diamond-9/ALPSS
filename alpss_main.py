from datetime import datetime
import traceback
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import os
from scipy.fft import (fft, ifft, fftfreq)
from scipy.fftpack import fftshift
from scipy.optimize import curve_fit
from scipy import signal
import findiff
import cv2 as cv
from scipy.signal import ShortTimeFFT


# main function to link together all the sub-functions
def alpss_main(**inputs):
    # attempt to run the program in full
    try:

        # begin the program timer
        start_time = datetime.now()

        # function to find the spall signal domain of interest
        sdf_out = spall_doi_finder(**inputs)

        # function to find the carrier frequency
        cen = carrier_frequency(sdf_out, **inputs)

        # function to filter out the carrier frequency after the signal has started
        cf_out = carrier_filter(sdf_out, cen, **inputs)

        # function to calculate the velocity from the filtered voltage signal
        vc_out = velocity_calculation(sdf_out, cen, cf_out, **inputs)

        # function to estimate the instantaneous uncertainty for all points in time
        iua_out = instantaneous_uncertainty_analysis(sdf_out, vc_out, cen, **inputs)

        # function to find points of interest on the velocity trace
        sa_out = spall_analysis(vc_out, iua_out, **inputs)

        # function to calculate uncertainties in the spall strength and strain rate due to external uncertainties
        fua_out = full_uncertainty_analysis(cen, sa_out, iua_out, **inputs)

        # end the program timer
        end_time = datetime.now()

        # function to generate the final figure
        fig = plotting(sdf_out, cen, cf_out, vc_out,
                       sa_out, iua_out, fua_out, start_time, end_time, **inputs)

        # function to save the output files if desired
        if inputs['save_data'] == 'yes':
            saving(sdf_out, cen, vc_out, sa_out, iua_out, fua_out, start_time, end_time, fig, **inputs)

        # end final timer and display full runtime
        end_time2 = datetime.now()
        print(f'\nFull program runtime (including plotting and saving):\n{end_time2 - start_time}\n')

    # in case the program throws an error
    except Exception:

        # print the traceback for the error
        print(traceback.format_exc())

        # attempt to plot the voltage signal from the imported data
        try:

            # import the desired data. Convert the time to skip and turn into number of rows
            t_step = 1 / inputs['sample_rate']
            rows_to_skip = inputs['header_lines'] + inputs['time_to_skip'] / t_step  # skip the header lines too
            nrows = inputs['time_to_take'] / t_step

            # change directory to where the data is stored
            os.chdir(inputs['exp_data_dir'])
            data = pd.read_csv(inputs['filename'], skiprows=int(rows_to_skip), nrows=int(nrows))

            # rename the columns of the data
            data.columns = ['Time', 'Ampl']

            # put the data into numpy arrays. Zero the time data
            time = data['Time'].to_numpy()
            time = time - time[0]
            voltage = data['Ampl'].to_numpy()

            # calculate the sample rate from the experimental data
            fs = 1 / np.mean(np.diff(time))

            # calculate the short time fourier transform
            f, t, Zxx = stft(voltage, fs, **inputs)

            # calculate magnitude of Zxx
            mag = np.abs(Zxx)

            # plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, num=2, figsize=(11, 4), dpi=300)
            ax1.plot(time / 1e-9, voltage / 1e-3)
            ax1.set_xlabel('Time (ns)')
            ax1.set_ylabel('Voltage (mV)')
            ax2.imshow(10 * np.log10(mag ** 2), aspect='auto', origin='lower',
                       interpolation='none', extent=[t[0] / 1e-9, t[-1] / 1e-9,
                                                     f[0] / 1e9, f[-1] / 1e9],
                       cmap=inputs['cmap'])
            ax2.set_xlabel('Time (ns)')
            ax2.set_ylabel('Frequency (GHz)')
            fig.suptitle('ERROR: Program Failed', c='r', fontsize=16)

            plt.tight_layout()
            plt.show()

        # if that also fails then print the traceback and stop running the program
        except Exception:
            print(traceback.format_exc())


# function to filter out the carrier frequency
def carrier_filter(sdf_out, cen, **inputs):
    # unpack dictionary values in to individual variables
    time = sdf_out['time']
    voltage = sdf_out['voltage']
    t_start_corrected = sdf_out['t_start_corrected']
    fs = sdf_out['fs']
    order = inputs['order']
    wid = inputs['wid']
    f_min = inputs['freq_min']
    f_max = inputs['freq_max']
    t_doi_start = sdf_out['t_doi_start']
    t_doi_end = sdf_out['t_doi_end']

    # get the index in the time array where the signal begins
    sig_start_idx = np.argmin(np.abs(time - t_start_corrected))

    # filter the data after the signal start time with a gaussian notch
    freq = fftshift(
        np.arange(-len(time[sig_start_idx:]) / 2, len(time[sig_start_idx:]) / 2) * fs / len(time[sig_start_idx:]))
    filt_2 = 1 - np.exp(-(freq - cen) ** order / wid ** order) - np.exp(-(freq + cen) ** order / wid ** order)
    voltage_filt = ifft(fft(voltage[sig_start_idx:]) * filt_2)

    # pair the filtered voltage from after the signal starts with the original data from before the signal starts
    voltage_filt = np.concatenate((voltage[0:sig_start_idx], voltage_filt))

    # perform a stft on the filtered voltage data. Only the real part as to not get a two sided spectrogram
    f_filt, t_filt, Zxx_filt = stft(np.real(voltage_filt), fs, **inputs)

    # calculate the power
    power_filt = 10 * np.log10(np.abs(Zxx_filt) ** 2)

    # cut the data to the domain of interest
    f_min_idx = np.argmin(np.abs(f_filt - f_min))
    f_max_idx = np.argmin(np.abs(f_filt - f_max))
    t_doi_start_idx = np.argmin(np.abs(t_filt - t_doi_start))
    t_doi_end_idx = np.argmin(np.abs(t_filt - t_doi_end))
    Zxx_filt_doi = Zxx_filt[f_min_idx:f_max_idx, t_doi_start_idx:t_doi_end_idx]
    power_filt_doi = power_filt[f_min_idx:f_max_idx, t_doi_start_idx:t_doi_end_idx]

    # save outputs to a dictionary
    cf_out = {
        'voltage_filt': voltage_filt,
        'f_filt': f_filt,
        't_filt': t_filt,
        'Zxx_filt': Zxx_filt,
        'power_filt': power_filt,
        'Zxx_filt_doi': Zxx_filt_doi,
        'power_filt_doi': power_filt_doi
    }

    return cf_out


# calculate the carrier frequency as the frequency with the max amplitude within the frequency range of interest
# specified in the user inputs
def carrier_frequency(spall_doi_finder_outputs, **inputs):
    # unpack dictionary values in to individual variables
    fs = spall_doi_finder_outputs['fs']
    time = spall_doi_finder_outputs['time']
    voltage = spall_doi_finder_outputs['voltage']
    freq_min = inputs['freq_min']
    freq_max = inputs['freq_max']

    # calculate frequency values for fft
    freq = fftfreq(int(fs * time[-1]) + 1, 1 / fs)
    freq2 = freq[:int(freq.shape[0] / 2) - 1]

    # find the frequency indices that mark the range of interest
    freq_min_idx = np.argmin(np.abs(freq2 - freq_min))
    freq_max_idx = np.argmin(np.abs(freq2 - freq_max))

    # find the amplitude values for the fft
    ampl = np.abs(fft(voltage))
    ampl2 = ampl[:int(freq.shape[0] / 2) - 1]

    # cut the frequency and amplitude to the range of interest
    freq3 = freq2[freq_min_idx: freq_max_idx]
    ampl3 = ampl2[freq_min_idx: freq_max_idx]

    # find the carrier as the frequency with the max amplitude
    cen = freq3[np.argmax(ampl3)]

    # return the carrier frequency
    return cen


# program to calculate the uncertainty in the spall strength and strain rate
def full_uncertainty_analysis(cen, sa_out, iua_out, **inputs):
    '''
    Based on the work of Mallick et al.

    Mallick, D.D., Zhao, M., Parker, J. et al. Laser-Driven Flyers and Nanosecond-Resolved Velocimetry for Spall Studies
    in Thin Metal Foils. Exp Mech 59, 611–628 (2019). https://doi.org/10.1007/s11340-019-00519-x
    '''

    # unpack dictionary values in to individual variables
    rho = inputs['density']
    C0 = inputs['C0']
    lam = inputs['lam']
    delta_rho = inputs['delta_rho']
    delta_C0 = inputs['delta_C0']
    delta_lam = inputs['delta_lam']
    theta = inputs['theta']
    delta_theta = inputs['delta_theta']
    delta_freq_tb = sa_out['peak_velocity_freq_uncert']
    delta_freq_td = sa_out['max_ten_freq_uncert']
    delta_time_c = iua_out['tau']
    delta_time_d = iua_out['tau']
    freq_tb = (sa_out['v_max_comp'] * 2) / lam + cen
    freq_td = (sa_out['v_max_ten'] * 2) / lam + cen
    time_c = sa_out['t_max_comp']
    time_d = sa_out['t_max_ten']

    # assuming time c is the same as time b
    freq_tc = freq_tb
    delta_freq_tc = delta_freq_tb

    # convert angles to radians
    theta = theta * (np.pi / 180)
    delta_theta = delta_theta * (np.pi / 180)

    # calculate the individual terms for spall uncertainty
    term1 = -0.5 * rho * C0 * (lam / 2) * np.tan(theta) * (1 / np.cos(theta)) * (freq_tb - freq_td) * delta_theta
    term2 = 0.5 * rho * C0 * (lam / (2 * np.cos(theta))) * delta_freq_tb
    term3 = -0.5 * rho * C0 * (lam / (2 * np.cos(theta))) * delta_freq_td
    term4 = 0.5 * rho * C0 * (1 / (2 * np.cos(theta))) * (freq_tb - freq_td) * delta_lam
    term5 = 0.5 * rho * (lam / (2 * np.cos(theta))) * (freq_tb - freq_td) * delta_C0
    term6 = 0.5 * C0 * (lam / (2 * np.cos(theta))) * (freq_tb - freq_td) * delta_rho

    # calculate spall uncertainty
    delta_spall = np.sqrt(term1 ** 2 +
                          term2 ** 2 +
                          term3 ** 2 +
                          term4 ** 2 +
                          term5 ** 2 +
                          term6 ** 2)

    # calculate the individual terms for strain rate uncertainty
    d_f = freq_tc - freq_td
    d_t = time_d - time_c
    term7 = (-lam / (4 * C0 ** 2 * np.cos(theta))) * (d_f / d_t) * delta_C0
    term8 = (1 / (4 * C0 * np.cos(theta))) * (d_f / d_t) * delta_lam
    term9 = ((lam * np.tan(theta)) / (4 * C0 * np.cos(theta))) * (d_f / d_t) * delta_theta
    term10 = (lam / (4 * C0 * np.cos(theta))) * (1 / d_t) * delta_freq_tc
    term11 = (-lam / (4 * C0 * np.cos(theta))) * (1 / d_t) * delta_freq_td
    term12 = (-lam / (4 * C0 * np.cos(theta))) * (d_f / d_t ** 2) * delta_time_c
    term13 = (lam / (4 * C0 * np.cos(theta))) * (d_f / d_t ** 2) * delta_time_d

    # calculate strain rate uncertainty
    delta_strain_rate = np.sqrt(term7 ** 2 +
                                term8 ** 2 +
                                term9 ** 2 +
                                term10 ** 2 +
                                term11 ** 2 +
                                term12 ** 2 +
                                term13 ** 2)

    # save outputs to a dictionary
    fua_out = {'spall_uncert': delta_spall,
               'strain_rate_uncert': delta_strain_rate}

    return fua_out


# general function for a sinusoid
def sin_func(x, a, b, c, d):
    return a * np.sin(2 * np.pi * b * x + c) + d


# get the indices for the upper and lower envelope of the voltage signal
# https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    '''
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    '''

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


# gaussian distribution
def gauss(x, amp, sigma, mu):
    f = (amp / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return f


# calculate the fwhm of a gaussian distribution
def fwhm(smoothing_window, smoothing_wid, smoothing_amp, smoothing_sigma, smoothing_mu, fs):
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


# function to estimate the instantaneous uncertainty for all points in time
def instantaneous_uncertainty_analysis(sdf_out, vc_out, cen, **inputs):
    # unpack needed variables
    lam = inputs['lam']
    smoothing_window = inputs['smoothing_window']
    smoothing_wid = inputs['smoothing_wid']
    smoothing_amp = inputs['smoothing_amp']
    smoothing_sigma = inputs['smoothing_sigma']
    smoothing_mu = inputs['smoothing_mu']
    fs = sdf_out['fs']
    time = sdf_out['time']
    time_f = vc_out['time_f']
    voltage_filt = vc_out['voltage_filt']
    time_start_idx = vc_out['time_start_idx']
    time_end_idx = vc_out['time_end_idx']
    carrier_band_time = inputs['carrier_band_time']

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
        popt, pcov = curve_fit(sin_func, time_cut, voltage_filt_early, p0=[0.1, cen, 0, 0])
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
    tau = fwhm(smoothing_window, smoothing_wid, smoothing_amp, smoothing_sigma, smoothing_mu, fs)
    freq_uncert_scaling = (1 / np.pi) * (np.sqrt(6 / (fs * (tau ** 3))))
    freq_uncert = inst_noise * freq_uncert_scaling
    vel_uncert = freq_uncert * (lam / 2)

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
        'tau': tau,
        'freq_uncert_scaling': freq_uncert_scaling,
        'freq_uncert': freq_uncert,
        'vel_uncert': vel_uncert
    }

    return iua_out


# function to take the numerical derivative of input array phas (central difference with a 9-point stencil).
# phas is padded so that after smoothing the final velocity trace matches the length of the domain of interest.
# this avoids issues with handling the boundaries in the derivative and later in smoothing.
# https://github.com/maroba/findiff/tree/master
def num_derivative(phas, window, time_start_idx, time_end_idx, fs):
    # set 8th order accuracy to get a 9-point stencil. can change the accuracy order if desired
    acc = 8

    # calculate how much padding is needed. half_space padding comes from the length of the smoothing window.
    half_space = int(np.floor(window / 2))
    pad = int(half_space + acc / 2)

    # get only the section of interest
    phas_pad = phas[time_start_idx - pad:time_end_idx + pad]

    # calculate the phase angle derivative
    ddt = findiff.FinDiff(0, 1 / fs, 1, acc=acc)
    dpdt_pad = ddt(phas_pad) * (1 / (2 * np.pi))

    # this is the hard coded 9-point central difference code. this can be used in case the findiff package ever breaks
    # dpdt_pad = np.zeros(phas_pad.shape)
    # for i in range(4, len(dpdt_pad) - 4):
    #     dpdt_pad[i] = ((1 / 280) * phas_pad[i - 4]
    #                    + (-4 / 105) * phas_pad[i - 3]
    #                    + (1 / 5) * phas_pad[i - 2]
    #                    + (-4 / 5) * phas_pad[i - 1]
    #                    + (4 / 5) * phas_pad[i + 1]
    #                    + (-1 / 5) * phas_pad[i + 2]
    #                    + (4 / 105) * phas_pad[i + 3]
    #                    + (-1 / 280) * phas_pad[i + 4]) \
    #                   * (fs / (2 * np.pi))

    # output both the padded and un-padded derivatives
    dpdt = dpdt_pad[pad:-pad]
    dpdt_pad = dpdt_pad[int(acc / 2):-int(acc / 2)]

    return dpdt, dpdt_pad


# function to generate the final figure
def plotting(sdf_out, cen, cf_out, vc_out, sa_out, iua_out, fua_out, start_time, end_time, **inputs):
    # create the figure and axes
    fig = plt.figure(num=1, figsize=inputs['plot_figsize'], dpi=inputs['plot_dpi'])
    ax1 = plt.subplot2grid((3, 5), (0, 0))  # voltage data
    ax2 = plt.subplot2grid((3, 5), (0, 1))  # noise distribution histogram
    ax3 = plt.subplot2grid((3, 5), (1, 0))  # imported voltage spectrogram
    ax4 = plt.subplot2grid((3, 5), (1, 1))  # thresholded spectrogram
    ax5 = plt.subplot2grid((3, 5), (2, 0))  # spectrogram of the ROI
    ax6 = plt.subplot2grid((3, 5), (2, 1))  # filtered spectrogram of the ROI
    ax7 = plt.subplot2grid((3, 5), (0, 2), colspan=2)  # voltage in the ROI
    ax8 = plt.subplot2grid((3, 5), (1, 2), colspan=2, rowspan=2)  # velocity overlaid with spectrogram
    ax9 = ax8.twinx()  # spectrogram overlaid with velocity
    ax10 = plt.subplot2grid((3, 5), (0, 4))  # noise fraction
    ax11 = ax10.twinx()  # velocity uncertainty
    ax12 = plt.subplot2grid((3, 5), (1, 4))  # velocity trace and spall points
    ax13 = plt.subplot2grid((3, 5), (2, 4), colspan=1, rowspan=1)  # results table

    # voltage data
    ax1.plot(sdf_out['time'] / 1e-9, sdf_out['voltage'] * 1e3, label='Original Signal', c='tab:blue')
    ax1.plot(sdf_out['time'] / 1e-9, np.real(vc_out['voltage_filt']) * 1e3, label='Filtered Signal', c='tab:orange')
    ax1.plot(iua_out['time_cut'] / 1e-9, iua_out['volt_fit'] * 1e3, label='Sine Fit', c='tab:green')
    ax1.axvspan(sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9, ymin=-1, ymax=1, color='tab:red',
                alpha=0.35, ec='none', label='ROI', zorder=4)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_xlim([sdf_out['time'][0] / 1e-9, sdf_out['time'][-1] / 1e-9])
    ax1.legend(loc='upper right')
    ax1.set_title('Voltage Data')

    # noise distribution histogram
    ax2.hist(iua_out['noise'] * 1e3, bins=50, rwidth=0.8)
    ax2.set_xlabel('Noise (mV)')
    ax2.set_ylabel('Counts')
    ax2.set_title('Voltage Noise')

    # imported voltage spectrogram and a rectangle to show the ROI
    plt3 = ax3.imshow(10 * np.log10(sdf_out['mag'] ** 2), aspect='auto', origin='lower',
                      interpolation='none',
                      extent=[sdf_out['t'][0] / 1e-9, sdf_out['t'][-1] / 1e-9,
                              sdf_out['f'][0] / 1e9, sdf_out['f'][-1] / 1e9],
                      cmap=inputs['cmap'])
    fig.colorbar(plt3, ax=ax3, label='Power (dBm)')
    anchor = [sdf_out['t_doi_start'] / 1e-9, sdf_out['f_doi'][0] / 1e9]
    width = sdf_out['t_doi_end'] / 1e-9 - sdf_out['t_doi_start'] / 1e-9
    height = sdf_out['f_doi'][-1] / 1e9 - sdf_out['f_doi'][0] / 1e9
    win = Rectangle(anchor,
                    width,
                    height,
                    edgecolor='r',
                    facecolor='none',
                    linewidth=0.75,
                    linestyle='-')
    ax3.add_patch(win)
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Frequency (GHz)')
    ax3.minorticks_on()
    ax3.set_title('Spectrogram Original Signal')

    # plotting the thresholded spectrogram on the ROI to show how the signal start time is found
    ax4.imshow(sdf_out['th3'], aspect='auto', origin='lower', interpolation='none',
               extent=[sdf_out['t'][0] / 1e-9, sdf_out['t'][-1] / 1e-9,
                       sdf_out['f_doi'][0] / 1e9, sdf_out['f_doi'][-1] / 1e9],
               cmap=inputs['cmap'])
    ax4.axvline(sdf_out['t_start_detected'] / 1e-9, ls='--', c='r')
    ax4.axvline(sdf_out['t_start_corrected'] / 1e-9, ls='-', c='r')
    if inputs['start_time_user'] == 'none':
        ax4.axhline(sdf_out['f_doi'][sdf_out['f_doi_carr_top_idx']] / 1e9, c='r')
    ax4.set_ylim([inputs['freq_min'] / 1e9, inputs['freq_max'] / 1e9])
    ax4.set_xlim([sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9])
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Frequency (GHz)')
    ax4.minorticks_on()
    ax4.set_title('Thresholded Spectrogram')

    # plotting the spectrogram of the ROI with the start-time line to see how well it lines up
    plt5 = ax5.imshow(10 * np.log10(sdf_out['mag'] ** 2), aspect='auto', origin='lower',
                      interpolation='none',
                      extent=[sdf_out['t'][0] / 1e-9, sdf_out['t'][-1] / 1e-9,
                              sdf_out['f'][0] / 1e9, sdf_out['f'][-1] / 1e9],
                      cmap=inputs['cmap'])
    fig.colorbar(plt5, ax=ax5, label='Power (dBm)')
    ax5.axvline(sdf_out['t_start_detected'] / 1e-9, ls='--', c='r')
    ax5.axvline(sdf_out['t_start_corrected'] / 1e-9, ls='-', c='r')
    if inputs['start_time_user'] == 'none':
        ax5.axhline(sdf_out['f_doi'][sdf_out['f_doi_carr_top_idx']] / 1e9, c='r')
    ax5.set_ylim([inputs['freq_min'] / 1e9, inputs['freq_max'] / 1e9])
    ax5.set_xlim([sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9])
    plt5.set_clim([np.min(sdf_out['power_doi']), np.max(sdf_out['power_doi'])])
    ax5.set_xlabel('Time (ns)')
    ax5.set_ylabel('Frequency (GHz)')
    ax5.minorticks_on()
    ax5.set_title('Spectrogram ROI')

    # plotting the filtered spectrogram of the ROI
    plt6 = ax6.imshow(cf_out['power_filt'], aspect='auto', origin='lower',
                      interpolation='none',
                      extent=[cf_out['t_filt'][0] / 1e-9, cf_out['t_filt'][-1] / 1e-9,
                              cf_out['f_filt'][0] / 1e9, cf_out['f_filt'][-1] / 1e9],
                      cmap=inputs['cmap'])
    fig.colorbar(plt6, ax=ax6, label='Power (dBm)')
    ax6.axvline(sdf_out['t_start_detected'] / 1e-9, ls='--', c='r')
    ax6.axvline(sdf_out['t_start_corrected'] / 1e-9, ls='-', c='r')
    ax6.set_ylim([inputs['freq_min'] / 1e9, inputs['freq_max'] / 1e9])
    ax6.set_xlim([sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9])
    plt6.set_clim([np.min(cf_out['power_filt_doi']), np.max(cf_out['power_filt_doi'])])
    ax6.set_xlabel('Time (ns)')
    ax6.set_ylabel('Frequency (GHz)')
    ax6.minorticks_on()
    ax6.set_title('Filtered Spectrogram ROI')

    # voltage in the ROI and the signal envelope
    ax7.plot(sdf_out['time'] / 1e-9, np.real(vc_out['voltage_filt']) * 1e3, label='Filtered Signal', c='tab:blue')
    ax7.plot(vc_out['time_f'] / 1e-9, iua_out['env_max_interp'] * 1e3, label='Signal Envelope', c='tab:red')
    ax7.plot(vc_out['time_f'] / 1e-9, iua_out['env_min_interp'] * 1e3, c='tab:red')
    ax7.set_xlabel('Time (ns)')
    ax7.set_ylabel('Voltage (mV)')
    ax7.set_xlim([sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9])
    ax7.legend(loc='upper right')
    ax7.set_title('Voltage ROI')

    # plotting the velocity and smoothed velocity curves to be overlaid on top of the spectrogram
    ax8.plot((vc_out['time_f']) / 1e-9,
             vc_out['velocity_f'], '-', c='grey', alpha=0.65, linewidth=3, label='Velocity')
    ax8.plot((vc_out['time_f']) / 1e-9,
             vc_out['velocity_f_smooth'], 'k-', linewidth=3, label='Smoothed Velocity')
    ax8.plot(vc_out['time_f'] / 1e-9, vc_out['velocity_f_smooth'] + iua_out['vel_uncert'] * inputs['uncert_mult'], 'r-',
             alpha=0.5,
             label=fr'$1\sigma$ Uncertainty (x{inputs["uncert_mult"]})')
    ax8.plot(vc_out['time_f'] / 1e-9, vc_out['velocity_f_smooth'] - iua_out['vel_uncert'] * inputs['uncert_mult'], 'r-',
             alpha=0.5)
    ax8.set_xlabel('Time (ns)')
    ax8.set_ylabel('Velocity (m/s)')
    ax8.legend(loc='lower right', fontsize=9, framealpha=1)
    ax8.set_zorder(1)
    ax8.patch.set_visible(False)
    ax8.set_title('Filtered Spectrogram ROI with Velocity')

    # plotting the final spectrogram to go with the velocity curves
    plt9 = ax9.imshow(cf_out['power_filt'],
                      extent=[cf_out['t_filt'][0] / 1e-9, cf_out['t_filt'][-1] / 1e-9,
                              cf_out['f_filt'][0] / 1e9, cf_out['f_filt'][-1] / 1e9],
                      aspect='auto',
                      origin='lower',
                      interpolation='none',
                      cmap=inputs['cmap'])
    ax9.set_ylabel('Frequency (GHz)')
    vel_lim = np.array([-300, np.max(vc_out['velocity_f_smooth']) + 300])
    ax8.set_ylim(vel_lim)
    ax8.set_xlim([cf_out['t_filt'][0] / 1e-9, cf_out['t_filt'][-1] / 1e-9])
    freq_lim = (vel_lim / (inputs['lam'] / 2)) + cen
    ax9.set_ylim(freq_lim / 1e9)
    ax9.set_xlim([sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9])
    ax9.minorticks_on()
    plt9.set_clim([np.min(cf_out['power_filt_doi']), np.max(cf_out['power_filt_doi'])])

    # plot the noise fraction on the ROI
    ax10.plot(vc_out['time_f'] / 1e-9, iua_out['inst_noise'] * 100, 'r', linewidth=2)
    ax10.set_xlabel('Time (ns)')
    ax10.set_ylabel('Noise Fraction (%)')
    ax10.set_xlim([vc_out['time_f'][0] / 1e-9, vc_out['time_f'][-1] / 1e-9])
    ax10.minorticks_on()
    ax10.grid(axis='both', which='both')
    ax10.set_title('Noise Fraction and Velocity Uncertainty')

    # plot the velocity uncertainty on the ROI
    ax11.plot(vc_out['time_f'] / 1e-9, iua_out['vel_uncert'], linewidth=2)
    ax11.set_ylabel('Velocity Uncertainty (m/s)')
    ax11.minorticks_on()

    # plotting the final smoothed velocity trace and uncertainty bounds with spall point markers (if they were found
    # on the signal)
    ax12.fill_between(
        (vc_out['time_f'] - sdf_out['t_start_corrected']) / 1e-9,
        vc_out['velocity_f_smooth'] + 2 * iua_out['vel_uncert'] * inputs['uncert_mult'],
        vc_out['velocity_f_smooth'] - 2 * iua_out['vel_uncert'] * inputs['uncert_mult'],
        color='mistyrose',
        label=fr'$2\sigma$ Uncertainty (x{inputs["uncert_mult"]})'
    )

    ax12.fill_between(
        (vc_out['time_f'] - sdf_out['t_start_corrected']) / 1e-9,
        vc_out['velocity_f_smooth'] + iua_out['vel_uncert'] * inputs['uncert_mult'],
        vc_out['velocity_f_smooth'] - iua_out['vel_uncert'] * inputs['uncert_mult'],
        color='lightcoral',
        alpha=0.5,
        ec='none',
        label=fr'$1\sigma$ Uncertainty (x{inputs["uncert_mult"]})'
    )

    ax12.plot((vc_out['time_f'] - sdf_out['t_start_corrected']) / 1e-9,
              vc_out['velocity_f_smooth'], 'k-', linewidth=3, label='Smoothed Velocity')
    ax12.set_xlabel('Time (ns)')
    ax12.set_ylabel('Velocity (m/s)')
    ax12.set_title('Velocity with Uncertainty Bounds')

    if not np.isnan(sa_out['t_max_comp']):
        ax12.plot((sa_out['t_max_comp'] - sdf_out['t_start_corrected']) / 1e-9, sa_out['v_max_comp'], 'bs',
                  label=f'Velocity at Max Compression: {int(round(sa_out["v_max_comp"]))}')
    if not np.isnan(sa_out['t_max_ten']):
        ax12.plot((sa_out['t_max_ten'] - sdf_out['t_start_corrected']) / 1e-9, sa_out['v_max_ten'], 'ro',
                  label=f'Velocity at Max Tension: {int(round(sa_out["v_max_ten"]))}')
    if not np.isnan(sa_out['t_rc']):
        ax12.plot((sa_out['t_rc'] - sdf_out['t_start_corrected']) / 1e-9, sa_out['v_rc'], 'gD',
                  label=f'Velocity at Recompression: {int(round(sa_out["v_rc"]))}')

    # if not np.isnan(sa_out['t_max_comp']) or not np.isnan(sa_out['t_max_ten']) or not np.isnan(sa_out['t_rc']):
    #    ax12.legend(loc='lower right', fontsize=9)
    ax12.legend(loc='lower right', fontsize=9)
    ax12.set_xlim([-inputs['t_before'] / 1e-9, (vc_out['time_f'][-1] - sdf_out['t_start_corrected']) / 1e-9])
    ax12.set_ylim([np.min(vc_out['velocity_f_smooth']) - 100, np.max(vc_out['velocity_f_smooth']) + 100])

    if np.max(iua_out['inst_noise']) > 1.0:
        ax10.set_ylim([0, 100])
        ax11.set_ylim([0, iua_out['freq_uncert_scaling'] * (inputs['lam'] / 2)])

    # table to show results of the run
    run_data1 = {'Name': ['Date',
                          'Time',
                          'File Name',
                          'Run Time',
                          'Smoothing FWHM (ns)',
                          'Peak Shock Stress (GPa)',
                          'Strain Rate (x1e6)',
                          'Spall Strength (GPa)'],
                 'Value': [start_time.strftime('%b %d %Y'),
                           start_time.strftime('%I:%M %p'),
                           inputs['filename'],
                           (end_time - start_time),
                           round(iua_out['tau'] * 1e9, 2),
                           round((.5 * inputs['density'] * inputs['C0'] * sa_out['v_max_comp']) / 1e9, 6),
                           fr"{round(sa_out['strain_rate_est'] / 1e6, 6)} $\pm$ {round(fua_out['strain_rate_uncert'] / 1e6, 6)}",
                           fr"{round(sa_out['spall_strength_est'] / 1e9, 6)} $\pm$ {round(fua_out['spall_uncert'] / 1e9, 6)}"]}

    df1 = pd.DataFrame(data=run_data1)
    cellLoc1 = 'center'
    loc1 = 'center'
    table1 = ax13.table(cellText=df1.values,
                        colLabels=df1.columns,
                        cellLoc=cellLoc1,
                        loc=loc1)
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)
    ax13.axis('tight')
    ax13.axis('off')

    # fix the layout
    plt.tight_layout()

    # display the plots if desired. if this is turned off the plots will still save
    if inputs['display_plots'] == 'yes':
        plt.show()

    # return the figure so it can be saved if desired
    return fig


# function for saving all the final outputs
def saving(sdf_out, cen, vc_out, sa_out, iua_out, fua_out, start_time, end_time, fig, **inputs):
    # change to the output files directory
    os.chdir(inputs['out_files_dir'])

    # save the plots
    fig.savefig(fname=(inputs['filename'][0:-4] + '--plots.png'), dpi='figure', format='png', facecolor='w')

    # save the function inputs used for this run
    inputs_df = pd.DataFrame.from_dict(inputs, orient='index', columns=['Input'])
    inputs_df.to_csv(inputs['filename'][0:-4] + '--inputs' + '.csv', index=True, header=False)

    # save the noisy velocity trace
    velocity_data = np.stack((vc_out['time_f'], vc_out['velocity_f']), axis=1)
    np.savetxt(inputs['filename'][0:-4] + '--velocity' + '.csv', velocity_data, delimiter=',')

    # save the smoothed velocity trace
    velocity_data_smooth = np.stack((vc_out['time_f'], vc_out['velocity_f_smooth']), axis=1)
    np.savetxt(inputs['filename'][0:-4] + '--velocity--smooth' + '.csv', velocity_data_smooth, delimiter=',')

    # save the filtered voltage data
    voltage_data = np.stack((sdf_out['time'], np.real(vc_out['voltage_filt']), np.imag(vc_out['voltage_filt'])), axis=1)
    np.savetxt(inputs['filename'][0:-4] + '--voltage' + '.csv', voltage_data, delimiter=',')

    # save the noise fraction
    noise_data = np.stack((vc_out['time_f'], iua_out['inst_noise']), axis=1)
    np.savetxt(inputs['filename'][0:-4] + '--noisefrac' + '.csv', noise_data, delimiter=',')

    # save the velocity uncertainty
    vel_uncert_data = np.stack((vc_out['time_f'], iua_out['vel_uncert']), axis=1)
    np.savetxt(inputs['filename'][0:-4] + '--veluncert' + '.csv', vel_uncert_data, delimiter=',')

    # save the final results
    results_to_save = {'Name': ['Date',
                                'Time',
                                'File Name',
                                'Run Time',
                                'Velocity at Max Compression',
                                'Time at Max Compression',
                                'Velocity at Max Tension',
                                'Time at Max Tension',
                                'Velocity at Recompression',
                                'Time at Recompression',
                                'Carrier Frequency',
                                'Spall Strength',
                                'Spall Strength Uncertainty',
                                'Strain Rate',
                                'Strain Rate Uncertainty',
                                'Peak Shock Stress',
                                'Spect Time Res',
                                'Spect Freq Res',
                                'Spect Velocity Res',
                                'Signal Start Time',
                                'Smoothing Characteristic Time'],
                       'Value': [start_time.strftime('%b %d %Y'),
                                 start_time.strftime('%I:%M %p'),
                                 inputs['filename'],
                                 (end_time - start_time),
                                 sa_out['v_max_comp'],
                                 sa_out['t_max_comp'],
                                 sa_out['v_max_ten'],
                                 sa_out['t_max_ten'],
                                 sa_out['v_rc'],
                                 sa_out['t_rc'],
                                 cen,
                                 sa_out['spall_strength_est'],
                                 fua_out['spall_uncert'],
                                 sa_out['strain_rate_est'],
                                 fua_out['strain_rate_uncert'],
                                 (.5 * inputs['density'] * inputs['C0'] * sa_out['v_max_comp']),
                                 sdf_out['t_res'],
                                 sdf_out['f_res'],
                                 0.5 * (inputs['lam'] * sdf_out['f_res']),
                                 sdf_out['t_start_corrected'],
                                 iua_out['tau']]}
    results_df = pd.DataFrame(data=results_to_save)
    results_df.to_csv(inputs['filename'][0:-4] + '--results' + '.csv', index=False, header=False)

    # display the final results table in nanoseconds to make it more readable
    # the data in the saved file is still in seconds
    results_df['Value'][5] = results_df['Value'][5] / 1e-9
    results_df['Value'][7] = results_df['Value'][7] / 1e-9
    results_df['Value'][9] = results_df['Value'][9] / 1e-9
    results_df['Value'][16] = results_df['Value'][16] / 1e-9
    results_df['Value'][19] = results_df['Value'][19] / 1e-9
    results_df['Value'][20] = results_df['Value'][20] / 1e-9
    display(results_df)


# function for smoothing the padded velocity data; padded data is used so the program can return
# a smooth velocity over the full domain of interest without running in to issues with the boundaries
def smoothing(velocity_pad, smoothing_window, smoothing_wid, smoothing_amp, smoothing_sigma, smoothing_mu):
    # if the smoothing window is not an odd integer exit the program
    if (smoothing_window % 2 != 1) or (smoothing_window >= len(velocity_pad) / 2):
        raise Exception('Input variable "smoothing_window" must be an odd integer and less than half the length of '
                        'the velocity signal')

    # number of points to either side of the point of interest
    half_space = int(np.floor(smoothing_window / 2))

    # weights to be applied to each sliding window as calculated from a normal distribution
    weights = gauss(np.linspace(-smoothing_wid, smoothing_wid, smoothing_window),
                    smoothing_amp, smoothing_sigma, smoothing_mu)

    # iterate over the domain and calculate the gaussian weighted moving average
    velocity_f_smooth = np.zeros(len(velocity_pad) - smoothing_window + 1)
    for i in range(half_space, len(velocity_f_smooth) + half_space):
        vel_pad_win = velocity_pad[i - half_space:i + half_space + 1]
        velocity_f_smooth[i - half_space] = np.average(vel_pad_win, weights=weights)

    # return the smoothed velocity
    return velocity_f_smooth


# function to pull out important points on the spall signal
def spall_analysis(vc_out, iua_out, **inputs):
    # if user wants to pull out the spall points
    if inputs['spall_calculation'] == 'yes':

        # unpack dictionary values in to individual variables
        time_f = vc_out['time_f']
        velocity_f_smooth = vc_out['velocity_f_smooth']
        pb_neighbors = inputs['pb_neighbors']
        pb_idx_correction = inputs['pb_idx_correction']
        rc_neighbors = inputs['pb_neighbors']
        rc_idx_correction = inputs['pb_idx_correction']
        C0 = inputs['C0']
        density = inputs['density']
        freq_uncert = iua_out['freq_uncert']
        vel_uncert = iua_out['vel_uncert']

        # get the global peak velocity
        peak_velocity_idx = np.argmax(velocity_f_smooth)
        peak_velocity = velocity_f_smooth[peak_velocity_idx]

        # get the uncertainities associated with the peak velocity
        peak_velocity_freq_uncert = freq_uncert[peak_velocity_idx]
        peak_velocity_vel_uncert = vel_uncert[peak_velocity_idx]

        # attempt to get the fist local minimum after the peak velocity to get the pullback
        # velocity. 'order' is the number of points on each side to compare to.
        try:

            # get all the indices for relative minima in the domain, order them, and take the first one that occurs
            # after the peak velocity
            rel_min_idx = signal.argrelmin(velocity_f_smooth, order=pb_neighbors)[0]
            extrema_min = np.append(rel_min_idx, np.argmax(velocity_f_smooth))
            extrema_min.sort()
            max_ten_idx = extrema_min[
                np.where(extrema_min == np.argmax(velocity_f_smooth))[0][0] + 1 + pb_idx_correction]

            # get the uncertainities associated with the max tension velocity
            max_ten_freq_uncert = freq_uncert[max_ten_idx]
            max_ten_vel_uncert = vel_uncert[max_ten_idx]

            # get the velocity at max tension
            max_tension_velocity = velocity_f_smooth[max_ten_idx]

            # calculate the pullback velocity
            pullback_velocity = peak_velocity - max_tension_velocity

            # calculate the estimated strain rate and spall strength
            strain_rate_est = (0.5 / C0) * pullback_velocity / (
                    time_f[max_ten_idx] - time_f[np.argmax(velocity_f_smooth)])
            spall_strength_est = 0.5 * density * C0 * pullback_velocity

            # set final variables for the function return
            t_max_comp = time_f[np.argmax(velocity_f_smooth)]
            t_max_ten = time_f[max_ten_idx]
            v_max_comp = peak_velocity
            v_max_ten = max_tension_velocity

        # if the program fails to find the peak and pullback velocities, then input nan's and continue with the program
        except Exception:
            print(traceback.format_exc())
            print('Could not locate the peak and/or pullback velocity')
            t_max_comp = np.nan
            t_max_ten = np.nan
            v_max_comp = np.nan
            v_max_ten = np.nan
            strain_rate_est = np.nan
            spall_strength_est = np.nan
            max_ten_freq_uncert = np.nan
            max_ten_vel_uncert = np.nan

        # try to get the recompression peak that occurs after pullback
        try:
            # get first local maximum after pullback
            rel_max_idx = signal.argrelmax(velocity_f_smooth, order=rc_neighbors)[0]
            extrema_max = np.append(rel_max_idx, np.argmax(velocity_f_smooth))
            extrema_max.sort()
            rc_idx = extrema_max[np.where(extrema_max == np.argmax(velocity_f_smooth))[0][0] + 2 + rc_idx_correction]
            t_rc = time_f[rc_idx]
            v_rc = velocity_f_smooth[rc_idx]

        # if finding the recompression peak fails then input nan's and continue
        except Exception:
            print(traceback.format_exc())
            print('Could not locate the recompression velocity')
            t_rc = np.nan
            v_rc = np.nan

    # if user does not want to pull out the spall points just set everything to nan
    else:
        t_max_comp = np.nan
        t_max_ten = np.nan
        t_rc = np.nan
        v_max_comp = np.nan
        v_max_ten = np.nan
        v_rc = np.nan
        spall_strength_est = np.nan
        strain_rate_est = np.nan
        peak_velocity_freq_uncert = np.nan
        peak_velocity_vel_uncert = np.nan
        max_ten_freq_uncert = np.nan
        max_ten_vel_uncert = np.nan

    # return a dictionary of the results
    sa_out = {
        't_max_comp': t_max_comp,
        't_max_ten': t_max_ten,
        't_rc': t_rc,
        'v_max_comp': v_max_comp,
        'v_max_ten': v_max_ten,
        'v_rc': v_rc,
        'spall_strength_est': spall_strength_est,
        'strain_rate_est': strain_rate_est,
        'peak_velocity_freq_uncert': peak_velocity_freq_uncert,
        'peak_velocity_vel_uncert': peak_velocity_vel_uncert,
        'max_ten_freq_uncert': max_ten_freq_uncert,
        'max_ten_vel_uncert': max_ten_vel_uncert
    }

    return sa_out


# function to find the specific domain of interest in the larger signal
def spall_doi_finder(**inputs):
    # import the desired data. Convert the time to skip and turn into number of rows
    t_step = 1 / inputs['sample_rate']
    rows_to_skip = inputs['header_lines'] + inputs['time_to_skip'] / t_step  # skip the 5 header lines too
    nrows = inputs['time_to_take'] / t_step

    # change directory to where the data is stored
    os.chdir(inputs['exp_data_dir'])
    data = pd.read_csv(inputs['filename'], skiprows=int(rows_to_skip), nrows=int(nrows))

    # rename the columns of the data
    data.columns = ['Time', 'Ampl']

    # put the data into numpy arrays. Zero the time data
    time = data['Time'].to_numpy()
    time = time - time[0]
    voltage = data['Ampl'].to_numpy()

    # calculate the true sample rate from the experimental data
    fs = 1 / np.mean(np.diff(time))

    # calculate the short time fourier transform
    f, t, Zxx = stft(voltage, fs, **inputs)

    # calculate magnitude of Zxx
    mag = np.abs(Zxx)

    # calculate the time and frequency resolution of the transform
    t_res = np.mean(np.diff(t))
    f_res = np.mean(np.diff(f))

    # find the index of the minimum and maximum frequencies as specified in the user inputs
    freq_min_idx = np.argmin(np.abs(f - inputs['freq_min']))
    freq_max_idx = np.argmin(np.abs(f - inputs['freq_max']))

    # cut the magnitude and frequency arrays to smaller ranges
    mag_cut = mag[freq_min_idx:freq_max_idx, :]
    f_doi = f[freq_min_idx:freq_max_idx]

    # calculate spectrogram power
    power_cut = 10 * np.log10(mag_cut ** 2)

    # convert spectrogram powers to uint8 for image processing
    smin = np.min(power_cut)
    smax = np.max(power_cut)
    a = 255 / (smax - smin)
    b = 255 - a * smax
    power_gray = a * power_cut + b
    power_gray8 = power_gray.astype(np.uint8)

    # blur using a gaussian filter
    blur = cv.GaussianBlur(power_gray8, inputs['blur_kernel'], inputs['blur_sigx'], inputs['blur_sigy'])

    # automated thresholding using Otsu's binarization
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # if not using a user input value for the signal start time
    if inputs['start_time_user'] == 'none':

        # Find the position/row of the top of the binary spectrogram for each time/column
        col_len = th3.shape[1]  # number of columns
        row_len = th3.shape[0]  # number of columns
        top_line = np.zeros(col_len)  # allocate space to place the indices
        f_doi_top_line = np.zeros(col_len)  # allocate space to place the corresponding frequencies

        for col_idx in range(col_len):  # loop over every column
            for row_idx in range(row_len):  # loop over every row

                # moving from the top down, if the pixel is 255 then store the index and break to move to the next column
                idx_top = row_len - row_idx - 1

                if th3[idx_top, col_idx] == 255:
                    top_line[col_idx] = idx_top
                    f_doi_top_line[col_idx] = f_doi[idx_top]
                    break

        # if the signal completely drops out there will be elements of f_doi_top_line equal to zero - these points are
        # made NaNs. Same for top_line.
        f_doi_top_line_clean = f_doi_top_line.copy()
        f_doi_top_line_clean[np.where(top_line == 0)] = np.nan
        top_line_clean = top_line.copy()
        top_line_clean[np.where(top_line == 0)] = np.nan

        # find the index of t where the time is closest to the user input carrier_band_time
        carr_idx = np.argmin(np.abs(t - inputs['carrier_band_time']))

        # calculate the average frequency of the top of the carrier band during carrier_band_time
        f_doi_carr_top_avg = np.mean(f_doi_top_line_clean[:carr_idx])

        # find the index in f_doi that is closest in frequency to f_doi_carr_top_avg
        f_doi_carr_top_idx = np.argmin(np.abs(f_doi - f_doi_carr_top_avg))

        # work backwards from the highest point on the signal top line until it matches or dips below f_doi_carr_top_idx
        highest_idx = np.argmax(f_doi_top_line_clean)
        for check_idx in range(highest_idx):
            cidx = highest_idx - check_idx - 1
            if top_line_clean[cidx] <= f_doi_carr_top_idx:
                break

        # add in the user correction for the start time
        t_start_detected = t[cidx]
        t_start_corrected = t_start_detected + inputs['start_time_correction']
        t_doi_start = t_start_corrected - inputs['t_before']
        t_doi_end = t_start_corrected + inputs['t_after']

        t_doi_start_spec_idx = np.argmin(np.abs(t - t_doi_start))
        t_doi_end_spec_idx = np.argmin(np.abs(t - t_doi_end))
        mag_doi = mag_cut[:, t_doi_start_spec_idx:t_doi_end_spec_idx]
        power_doi = 10 * np.log10(mag_doi ** 2)


    # if using a user input for the signal start time
    else:

        # these params become nan because they are only needed if the program
        # is finding the signal start time automatically
        f_doi_top_line_clean = np.nan
        carr_idx = np.nan
        f_doi_carr_top_idx = np.nan

        # use the user input signal start time to define the domain of interest
        t_start_detected = t[np.argmin(np.abs(t - inputs['start_time_user']))]
        t_start_corrected = t_start_detected + inputs['start_time_correction']
        t_doi_start = t_start_corrected - inputs['t_before']
        t_doi_end = t_start_corrected + inputs['t_after']

        t_doi_start_spec_idx = np.argmin(np.abs(t - t_doi_start))
        t_doi_end_spec_idx = np.argmin(np.abs(t - t_doi_end))
        mag_doi = mag_cut[:, t_doi_start_spec_idx:t_doi_end_spec_idx]
        power_doi = 10 * np.log10(mag_doi ** 2)

    # dictionary to return outputs
    sdf_out = {
        'time': time,
        'voltage': voltage,
        'fs': fs,
        'f': f,
        't': t,
        'Zxx': Zxx,
        't_res': t_res,
        'f_res': f_res,
        'f_doi': f_doi,
        'mag': mag,
        'th3': th3,
        'f_doi_top_line_clean': f_doi_top_line_clean,
        'carr_idx': carr_idx,
        'f_doi_carr_top_idx': f_doi_carr_top_idx,
        't_start_detected': t_start_detected,
        't_start_corrected': t_start_corrected,
        't_doi_start': t_doi_start,
        't_doi_end': t_doi_end,
        'power_doi': power_doi
    }

    return sdf_out


# function to calculate the short time fourier transform (stft) of a signal. ALPSS was originally built with a scipy
# STFT function that may now be deprecated in the future. This function seeks to roughly replicate the behavior of the
# legacy stft function, specifically how the time windows are calculated and how the boundaries are handled
def stft(voltage, fs, **inputs):
    # calculate stft with the new scipy library function and zero padding the boundaries
    SFT = ShortTimeFFT.from_window(inputs['window'], fs=fs, nperseg=inputs['nperseg'], noverlap=inputs['noverlap'],
                                   mfft=inputs['nfft'], scale_to='magnitude', phase_shift=None)
    Sx_full = SFT.stft(voltage, padding='zeros')
    t_full = SFT.t(len(voltage))
    f = SFT.f

    # calculate the time array for the legacy scipy stft function without zero padding on the boundaries
    t_legacy = np.arange(inputs['nperseg'] / 2, voltage.shape[-1] - inputs['nperseg'] / 2 + 1,
                         inputs['nperseg'] - inputs['noverlap']) / float(fs)

    # find the time index in the new stft function that corresponds to where the legacy function time array begins
    t_idx = np.argmin(np.abs(t_full - t_legacy[0]))

    # crop the time array to the length of the legacy function
    t_crop = t_full[t_idx: t_idx + len(t_legacy)]

    # crop the stft magnitude array to the length of the legacy function
    Sx_crop = Sx_full[:, t_idx: t_idx + len(t_legacy)]

    # return the frequency, time, and magnitude arrays
    return f, t_crop, Sx_crop


# function to calculate the velocity from the filtered voltage signal
def velocity_calculation(spall_doi_finder_outputs, cen, carrier_filter_outputs, **inputs):
    # unpack dictionary values in to individual variables
    fs = spall_doi_finder_outputs['fs']
    time = spall_doi_finder_outputs['time']
    voltage_filt = carrier_filter_outputs['voltage_filt']
    freq_min = inputs['freq_min']
    freq_max = inputs['freq_max']
    lam = inputs['lam']
    t_doi_start = spall_doi_finder_outputs['t_doi_start']
    t_doi_end = spall_doi_finder_outputs['t_doi_end']

    # isolate signal. filter out all frequencies that are outside the range of interest
    numpts = len(time)
    freq = fftshift(np.arange((-numpts / 2), (numpts / 2)) * fs / numpts)
    filt = (freq > freq_min) * (freq < freq_max)
    voltage_filt = ifft(fft(voltage_filt) * filt)

    # get the indices in the time array closest to the domain start and end times
    time_start_idx = np.argmin(np.abs(time - t_doi_start))
    time_end_idx = np.argmin(np.abs(time - t_doi_end))

    # unwrap the phase angle of the filtered voltage signal
    phas = np.unwrap(np.angle(voltage_filt), axis=0)

    # take the numerical derivative using the certral difference method with a 9-point stencil
    # return the derivative on the domain of interest (dpdt) as well as the padded derivative to be used for smoothing
    dpdt, dpdt_pad = num_derivative(phas, inputs['smoothing_window'], time_start_idx, time_end_idx, fs)

    # convert the derivative in to velocity
    velocity_pad = (lam / 2) * (dpdt_pad - cen)
    velocity_f = (lam / 2) * (dpdt - cen)

    # crop the time array
    time_f = time[time_start_idx:time_end_idx]

    # smooth the padded velocity signal using a moving average with gaussian weights
    velocity_f_smooth = smoothing(velocity_pad=velocity_pad,
                                  smoothing_window=inputs['smoothing_window'],
                                  smoothing_wid=inputs['smoothing_wid'],
                                  smoothing_amp=inputs['smoothing_amp'],
                                  smoothing_sigma=inputs['smoothing_sigma'],
                                  smoothing_mu=inputs['smoothing_mu'])

    # return a dictionary of the outputs
    vc_out = {
        'time_f': time_f,
        'velocity_f': velocity_f,
        'velocity_f_smooth': velocity_f_smooth,
        'phasD2_f': dpdt,
        'voltage_filt': voltage_filt,
        'time_start_idx': time_start_idx,
        'time_end_idx': time_end_idx
    }

    return vc_out
