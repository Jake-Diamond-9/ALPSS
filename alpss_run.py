"""
ALPSS: AnaLysis of Pdv Spall Signals
Jake Diamond (2022)
Johns Hopkins University
Hopkins Extreme Materials Institute (HEMI)
Please report any bugs or comments to jdiamo15@jhu.edu


Key for input variables:
filename:                   string; filename for the data to run
save_data:                  string; 'yes' or 'no' to save output data
start_time_user:            string or float; if 'none' the program will attempt to find the
                                             signal start time automatically. if float then
                                             the program will use that as the signal start time
spacer_thickness:           float; how far the sample will be placed from the flyer
impact_vel_averaging_dist:  float; distance over which to average the estimated impact velocity
header_lines:               integer; number of header lines to skip in the data file
time_to_skip:               float; the amount of time to skip in the full data file before beginning to read in data
time_to_take:               float; the amount of time to take in the data file after skipping time_to_skip
t_before:                   float; amount of time before the signal start time to include in the velocity calculation
t_after:                    float; amount of time after the signal start time to include in the velocity calculation
start_time_correction:      float; amount of time to adjust the signal start time by
freq_min:                   float; minimum frequency for the region of interest
freq_max:                   float; maximum frequency for the region of interest
smoothing_window:           int; number of points to use for the smoothing window. must be an odd number
smoothing_wid:              float; half the width of the normal distribution used
                                   to calculate the smoothing weights (recommend 3)
smoothing_amp:              float; amplitude of the normal distribution used to calculate
                                   the smoothing weights (recommend 1)
smoothing_sigma:            float; standard deviation of the normal distribution used
                                   to calculate the smoothing weights (recommend 1)
smoothing_mu:               float; mean of the normal distribution used to calculate
                                   the smoothing weights (recommend 0)
sample_rate:                float; sample rate of the oscilloscope used in the experiment
nperseg:                    integer; number of points to use per segment of the stft
noverlap:                   integer; number of points to overlap per segment of the stft
nfft:                       integer; number of points to zero pad per segment of the stft
window:                     string or tuple or array_like; window function to use for the stft (recommend 'hann')
blur_kernel:                tuple; kernel size for gaussian blur smoothing (recommend (5, 5))
blur_sigx:                  float; standard deviation of the gaussian blur kernel in the x direction (recommend 0)
blur_sigy:                  float; standard deviation of the gaussian blur kernel in the y direction (recommend 0)
carrier_band_time:          float; length of time from the beginning of the imported data window to average
                                   the frequency of the top of the carrier band in the thresholded spectrogram
cmap:                       string; colormap for the spectrograms (recommend 'viridis')
order:                      integer; order for the gaussian notch filter used to remove the carrier band (recommend 6)
wid:                        float; width of the gaussian notch filter used to remove the carrier band (recommend 1e8)
lam:                        float; wavelength of the target laser
delta_time_d:               float; uncertainty in time d
exp_data_dir:               string; directory from which to read the experimental data file
out_files_dir:              string; directory to save output data to
display_plots:              string; 'yes' to display the final plots and 'no' to not display them. if save_data='yes'
                                     and and display_plots='no' the plots will be saved but not displayed
plot_figsize:               tuple; figure size for the final plots
plot_dpi:                   float; dpi for the final plots
"""

from alpss_main import *


alpss_main(filename="F2--20220624--00092.txt",
           save_data='yes',
           start_time_user=1220e-9,
           spacer_thickness=245e-6,
           impact_vel_averaging_dist=5e-6,
           header_lines=5,
           time_to_skip=70e-6,
           time_to_take=3e-6,
           t_before=100e-9,
           t_after=700e-9,
           start_time_correction=0e-9,
           freq_min=2e9,
           freq_max=4.5e9,
           smoothing_window=801,
           smoothing_wid=3,
           smoothing_amp=1,
           smoothing_sigma=1,
           smoothing_mu=0,
           sample_rate=80e9,
           nperseg=512,
           noverlap=435,
           nfft=5120,
           window='hann',
           blur_kernel=(5, 5),
           blur_sigx=0,
           blur_sigy=0,
           carrier_band_time=250e-9,
           cmap='viridis',
           order=6,
           wid=1e8,
           lam=1547.461e-9,
           exp_data_dir="/Users/jakediamond/Desktop/Hopkins School Work/HEMI Research/Project 1 - Dynamic Cavitation in Polymers/PC Data (Annealed)/PDV",
           out_files_dir="/Users/jakediamond/Desktop/Hopkins School Work/HEMI Research/Project 1 - Dynamic Cavitation in Polymers/PC Data (Annealed)/Data_Analysis",
           display_plots='yes',
           plot_figsize=(12, 10),
           plot_dpi=300)
