import numpy as np
from scipy.fft import fft
from scipy.fft import ifft
from scipy.fftpack import fftshift
from alpss.utils import stft


# function to filter out the carrier frequency
def carrier_filter(sdf_out, cen, **inputs):
    # unpack dictionary values in to individual variables
    time = sdf_out["time"]
    voltage = sdf_out["voltage"]
    t_start_corrected = sdf_out["t_start_corrected"]
    fs = sdf_out["fs"]
    order = inputs["order"]
    wid = inputs["wid"]
    f_min = inputs["freq_min"]
    f_max = inputs["freq_max"]
    t_doi_start = sdf_out["t_doi_start"]
    t_doi_end = sdf_out["t_doi_end"]

    # get the index in the time array where the signal begins
    sig_start_idx = np.argmin(np.abs(time - t_start_corrected))

    # filter the data after the signal start time with a gaussian notch
    freq = fftshift(
        np.arange(-len(time[sig_start_idx:]) / 2, len(time[sig_start_idx:]) / 2)
        * fs
        / len(time[sig_start_idx:])
    )
    filt_2 = (
        1
        - np.exp(-((freq - cen) ** order) / wid**order)
        - np.exp(-((freq + cen) ** order) / wid**order)
    )
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
        "voltage_filt": voltage_filt,
        "f_filt": f_filt,
        "t_filt": t_filt,
        "Zxx_filt": Zxx_filt,
        "power_filt": power_filt,
        "Zxx_filt_doi": Zxx_filt_doi,
        "power_filt_doi": power_filt_doi,
    }

    return cf_out
