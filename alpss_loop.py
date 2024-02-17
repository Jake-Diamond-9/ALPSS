# script to loop over multiple files and run them with previously used input parameters

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

# loop over all the files
for i, file in enumerate(df_tally.Filename[0:10]):
    # start timer for single file
    fstart = datetime.now()

    # extract the filename and get the inputs for when that pdv data was processed by ALPSS
    fname = file[0:-len('--plots.png')]
    inp_name = results_path + '/' + fname + '--inputs.csv'
    inp_df = pd.read_csv(inp_name, index_col=0, names=['Value'])

    # print(inp_df)

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
               smoothing_window=int(inp_df.loc['smoothing_window', 'Value']),
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
               out_files_dir="/Users/jakediamond/Desktop/Hopkins School Work/HEMI Research/Project 2 - High Throughput Testing/ALPSS/Data_Analysis/ECAE_ALPSS_Results_500neighbors_w_uncertainty",
               display_plots='no',
               plot_figsize=(30, 10),
               plot_dpi=100)

    print(f"Completed {i + 1}/{len(df_tally.Filename)},    Run Time: {datetime.now() - fstart}")

print(f'Full Program Run Time: {datetime.now() - pstart}')
