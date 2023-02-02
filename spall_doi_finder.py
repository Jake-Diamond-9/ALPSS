import os
import pandas as pd
import numpy as np
from scipy import signal
import cv2 as cv


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
    f, t, Zxx = signal.stft(voltage,
                            fs=fs,
                            window=inputs['window'],
                            nperseg=inputs['nperseg'],
                            noverlap=inputs['noverlap'],
                            nfft=inputs['nfft'],
                            boundary=None)

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
    power_cut = 20 * np.log10(mag_cut ** 2)

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
        power_doi = 20 * np.log10(mag_doi ** 2)


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
        power_doi = 20 * np.log10(mag_doi ** 2)

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
