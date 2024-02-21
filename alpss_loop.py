# script to loop over multiple files and run them with previously used input parameters
'''
from alpss_main import *
import pandas as pd
from datetime import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# start full program timer
pstart = datetime.now()

# get the data from the excel sheet with the tallied ALPSS results
tally_path = '/Users/jakediamond/Desktop/Hopkins School Work/HEMI Research/Project 2 - High Throughput Testing/ALPSS/Data_Analysis/ECAE_tally_500neighbors.xlsx'
df_tally = pd.read_excel(tally_path, index_col=0)

# path to the pdv analysis results for all the files used in the ALPSS paper
results_path = '/Users/jakediamond/Desktop/Hopkins School Work/HEMI Research/Project 2 - High Throughput Testing/ALPSS/Data_Analysis/ECAE_ALPSS_Results_500neighbors'

# directory to store the output results
out_dir = "/Users/jakediamond/Desktop/Hopkins School Work/HEMI Research/Project 2 - High Throughput Testing/ALPSS/Data_Analysis/smoothing_study"

# path to save the results table
table_path = "/Users/jakediamond/Desktop/Hopkins School Work/HEMI Research/Project 2 - High Throughput Testing/ALPSS/Data_Analysis"

# list of smoothing parameters to check
#windows = np.array([101, 201, 301, 401, 451, 501, 551, 601, 701, 801, 901, 1001, 1101, 1201])
windows = np.array([1201])

# preallocate storage for a table that will hold the results for the spall strengths for different smoothing params
results = np.zeros([df_tally.shape[0], windows.shape[0]])

# calculate total number of runs
num_run = df_tally.shape[0] * windows.shape[0]

# total run counter
runs = 0

# variable to store run times
run_times = np.array([])

# loop over all smoothing window lengths
for j, win in enumerate(windows):

    # loop over all the files
    for i, file in enumerate(df_tally.Filename):
        # start timer for single file
        fstart = datetime.now()

        # extract the filename and get the inputs for when that pdv data was processed by ALPSS
        fname = file[0:-len('--plots.png')]
        inp_name = results_path + '/' + fname + '--inputs.csv'
        inp_df = pd.read_csv(inp_name, index_col=0, names=['Value'])

        # run the new uncertainty version of alpss with the same previous inputs
        alpss_main(filename=inp_df.loc['filename', 'Value'],
                   save_data='yes',
                   start_time_user=inp_df.loc['start_time_user', 'Value'],
                   header_lines=int(inp_df.loc['header_lines', 'Value']),
                   time_to_skip=float(inp_df.loc['time_to_skip', 'Value']),
                   time_to_take=float(inp_df.loc['time_to_take', 'Value']),
                   t_before=float(inp_df.loc['t_before', 'Value']),
                   t_after=float(inp_df.loc['t_after', 'Value']),
                   start_time_correction=float(inp_df.loc['start_time_correction', 'Value']),
                   freq_min=float(inp_df.loc['freq_min', 'Value']),
                   freq_max=float(inp_df.loc['freq_max', 'Value']),
                   smoothing_window=int(win),
                   smoothing_wid=float(inp_df.loc['smoothing_wid', 'Value']),
                   smoothing_amp=float(inp_df.loc['smoothing_amp', 'Value']),
                   smoothing_sigma=float(inp_df.loc['smoothing_sigma', 'Value']),
                   smoothing_mu=float(inp_df.loc['smoothing_mu', 'Value']),
                   pb_neighbors=int(inp_df.loc['pb_neighbors', 'Value']),
                   pb_idx_correction=int(inp_df.loc['pb_idx_correction', 'Value']),
                   rc_neighbors=int(inp_df.loc['rc_neighbors', 'Value']),
                   rc_idx_correction=int(inp_df.loc['rc_idx_correction', 'Value']),
                   sample_rate=float(inp_df.loc['sample_rate', 'Value']),
                   nperseg=int(inp_df.loc['nperseg', 'Value']),
                   noverlap=int(inp_df.loc['noverlap', 'Value']),
                   nfft=int(inp_df.loc['nfft', 'Value']),
                   window=inp_df.loc['window', 'Value'],
                   blur_kernel=(5, 5),
                   blur_sigx=float(inp_df.loc['blur_sigx', 'Value']),
                   blur_sigy=float(inp_df.loc['blur_sigy', 'Value']),
                   carrier_band_time=float(inp_df.loc['carrier_band_time', 'Value']),
                   cmap=inp_df.loc['cmap', 'Value'],
                   order=int(inp_df.loc['order', 'Value']),
                   wid=float(inp_df.loc['wid', 'Value']),
                   lam=float(inp_df.loc['lam', 'Value']),
                   C0=float(inp_df.loc['C0', 'Value']),
                   density=float(inp_df.loc['density', 'Value']),
                   delta_rho=float(inp_df.loc['delta_rho', 'Value']),
                   delta_C0=float(inp_df.loc['delta_C0', 'Value']),
                   delta_lam=float(inp_df.loc['delta_lam', 'Value']),
                   theta=float(inp_df.loc['theta', 'Value']),
                   delta_theta=float(inp_df.loc['delta_theta', 'Value']),
                   delta_freq_tb=float(inp_df.loc['delta_freq_tb', 'Value']),
                   delta_freq_td=float(inp_df.loc['delta_freq_td', 'Value']),
                   delta_time_c=float(inp_df.loc['delta_time_c', 'Value']),
                   delta_time_d=float(inp_df.loc['delta_time_d', 'Value']),
                   exp_data_dir=inp_df.loc['exp_data_dir', 'Value'],
                   out_files_dir=out_dir,
                   display_plots='no',
                   plot_figsize=(30, 10),
                   plot_dpi=50)

        # extract the spall strength and save to the table
        res_dir = out_dir
        res_str = res_dir + '/' + fname + '--results.csv'
        res_df = pd.read_csv(res_str, index_col=0, names=['Value'])
        results[i, j] = float(res_df.loc['Spall Strength', 'Value'])

        # update run counter
        runs += 1

        # calculate iteration run time
        ftime = datetime.now() - fstart

        # append to the full list of run times
        run_times = np.append(run_times, ftime)

        # calculate number of runs remaining
        runs_remaining = num_run - ((i+1)*(j+1))

        # calculate estimated time remaining
        time_remaining = runs_remaining * np.mean(run_times)

        print(f"Completed {runs}/{num_run}, Run Time: {ftime}, Estimated Time Remaining: {time_remaining}, Estimated Finish Time: {datetime.now() + time_remaining}")

    # write the results to an excel file after each smoothing window has been completed
    df_results_table = pd.DataFrame(results, columns=windows)
    results_table_path = table_path + '/smoothing_study_results1201.xlsx'
    #df_results_table.to_excel(results_table_path, index=False)

print(f'Full Program Run Time: {datetime.now() - pstart}')
'''

# script to plot the average spall strength as a function of smoothing window duration

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# get the tallied ecae data
tally_df = pd.read_excel(
    '/Users/jakediamond/Desktop/Hopkins School Work/HEMI Research/Project 2 - High Throughput Testing/ALPSS/Data_Analysis/ECAE_tally_500neighbors.xlsx',
    index_col=0)

# get th spall strength results of the smoothing study
res_df = pd.read_excel(
    '/Users/jakediamond/Desktop/Hopkins School Work/HEMI Research/Project 2 - High Throughput Testing/ALPSS/Data_Analysis/smoothing_study_results.xlsx')

# combine the dataframes
df = pd.concat([tally_df, res_df], axis=1)

# only take the runs that found the correct spall strengths for the initial ecae runs
df = df[(df['Success'] == 2) | (df['Success'] == 3)]

# drop the filename and success columns to do averaging
df = df.drop(['Filename', 'Success', 101, 201, 301, 401, 451, 501, 551], axis=1)

# convert results to numpy arrays to find the average and std
res = df.to_numpy()
windows = df.columns.to_numpy()
ss_mean = np.nanmean(res, axis=0)
ss_std = np.nanstd(res, axis=0)

# plot results
fig, ax = plt.subplots(1, 1)
#ax.errorbar(windows*(1/80), ss_mean/1e9, yerr=ss_std/1e9, ls='-', marker='o', c='k')
ax.plot(windows * (1 / 80), ss_mean / 1e9, 'ko-')
ax.set_xlabel('Smoothing Window (ns)')
ax.set_ylabel('Average Spall Strength (GPa)')
ax.set_ylim([1.1, 1.35])
plt.tight_layout()
plt.show()

# script to calculate the average noise seen in successfully processed signals.
'''
# key for tallied data: Correct Time [1], Correct Spall [2], Both [3], Failed [4]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# get the data from the excel sheet with the tallied ALPSS results
tally_path = '/Users/jakediamond/Desktop/Hopkins School Work/HEMI Research/Project 2 - High Throughput Testing/ALPSS/Data_Analysis/ECAE_tally_500neighbors.xlsx'
df_tally = pd.read_excel(tally_path, index_col=0)

# path to the pdv analysis results for all the files calculated with uncertainty
results_path = '/Users/jakediamond/Desktop/Hopkins School Work/HEMI Research/Project 2 - High Throughput Testing/ALPSS/Data_Analysis/ECAE_ALPSS_Results_500neighbors_w_uncertainty'

# list to store the extracted noise data
noise_list = np.array([])

# list to store the medians of the individual signals
median_list = np.array([])

# variable to store the name of the best signal
best_sig = ''

# count the number of files processed
j = 0

# loop over all the files
for i, file in enumerate(df_tally.Filename):

    if df_tally.loc[i, 'Success'] == 3:
        j += 1
        # extract the filename and get the results file
        fname = file[0:-len('--plots.png')]
        res_name = results_path + '/' + fname + '--results.csv'
        res_df = pd.read_csv(res_name, index_col=0, names=['Value'])

        # get the signal start time from the results
        t_start = float(res_df.loc['Signal Start Time', 'Value'])

        # get the spall time from the results
        t_spall = float(res_df.loc['Time at Max Tension', 'Value'])

        # read in the noise data
        noise_name = results_path + '/' + fname + '--noisefrac.csv'
        noise_df = pd.read_csv(noise_name, names=['time', 'noise'])
        time = noise_df['time'].to_numpy()
        noise = noise_df['noise'].to_numpy()

        # find the index nearest to the start and spall times
        t_start_idx = np.argmin(np.abs(t_start - time))
        t_spall_idx = np.argmin(np.abs(t_spall - time))

        # pull out the noise data for the time from t_start to t_spall and store it
        noise_cut = noise[t_start_idx: t_spall_idx]
        noise_list = np.append(noise_list, noise_cut)

        # calculate the median noise for the individual signal
        sig_med = np.median(noise_cut)
        median_list = np.append(median_list, sig_med)

        if sig_med <= np.min(median_list):
            best_sig = file


print(f'Median: {np.median(noise_list)}')
print(f'Mean: {np.mean(noise_list)}')
print(f'Standard Deviation: {np.std(noise_list)}')
print(f'Average of the Medians: {np.mean(median_list)}')
print(f'Standard Deviation of the Medians: {np.std(median_list)}')
print(f'Best Signal, {best_sig}, had a median noise fraction of {np.min(median_list)}')

# fig, ax = plt.subplots(1, 1)
# ax.violinplot(median_list)
# plt.tight_layout()
# plt.show()
'''
