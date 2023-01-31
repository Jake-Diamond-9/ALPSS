import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import pandas as pd

# function to generate the final figure
def plotting(sdf_out, cen, cf_out, vc_out, sa_out, fua_out, start_time, end_time, **inputs):

    # create the figure and axes
    fig = plt.figure(num=1, figsize=inputs['plot_figsize'], dpi=inputs['plot_dpi'])
    ax1 = plt.subplot2grid((4, 2), (0, 0))                  # imported data
    ax2 = plt.subplot2grid((4, 2), (0, 1))                  # thresholded spectrogram
    ax3 = plt.subplot2grid((4, 2), (1, 0))                  # spectrogram ROI
    ax4 = plt.subplot2grid((4, 2), (1, 1))                  # filtered spectrogram ROI
    ax5 = plt.subplot2grid((4, 2), (2, 0))                  # velocity overlaid with spectrogram
    ax6 = ax5.twinx()                                       # spectrogram overlaid with velocity
    ax7 = plt.subplot2grid((4, 2), (2, 1))                  # velocity and spall points
    ax8 = plt.subplot2grid((4, 2), (3, 0), colspan=1)       # table 1
    ax9 = plt.subplot2grid((4, 2), (3, 1), colspan=1)       # table 2

    # plotting the imported data and a rectangle to show the ROI
    plt1 = ax1.imshow(20 * np.log10(sdf_out['mag'] ** 2), aspect='auto', origin='lower',
                      interpolation='none',
                      extent=[sdf_out['t'][0] / 1e-9, sdf_out['t'][-1] / 1e-9,
                              sdf_out['f'][0] / 1e9, sdf_out['f'][-1] / 1e9],
                      cmap=inputs['cmap'])
    fig.colorbar(plt1, ax=ax1, label='Power (dB)')
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
    ax1.add_patch(win)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Frequency (GHz)')
    ax1.set_title('Imported Data')
    ax1.minorticks_on()

    # plotting the thresholded spectrogram on the ROI to show how the signal start time is found
    ax2.imshow(sdf_out['th3'], aspect='auto', origin='lower', interpolation='none',
               extent=[sdf_out['t'][0] / 1e-9, sdf_out['t'][-1] / 1e-9,
                       sdf_out['f_doi'][0] / 1e9, sdf_out['f_doi'][-1] / 1e9],
               cmap=inputs['cmap'])
    ax2.axvline(sdf_out['t_start_detected'] / 1e-9, ls='--', c='r')
    ax2.axvline(sdf_out['t_start_corrected'] / 1e-9, ls='-', c='r')
    if inputs['start_time_user'] == 'none':
        ax2.axhline(sdf_out['f_doi'][sdf_out['f_doi_carr_top_idx']] / 1e9, c='r')
    ax2.set_ylim([inputs['freq_min'] / 1e9, inputs['freq_max'] / 1e9])
    ax2.set_xlim([sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9])
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Frequency (GHz)')
    ax2.set_title('Thresholded Spectrogram')
    ax2.minorticks_on()

    # plotting the spectrogram ROI with the start time line to see how well it lines up
    plt3 = ax3.imshow(20 * np.log10(sdf_out['mag'] ** 2), aspect='auto', origin='lower',
                      interpolation='none',
                      extent=[sdf_out['t'][0] / 1e-9, sdf_out['t'][-1] / 1e-9,
                              sdf_out['f'][0] / 1e9, sdf_out['f'][-1] / 1e9],
                      cmap=inputs['cmap'])
    fig.colorbar(plt3, ax=ax3, label='Power (dB)')
    ax3.axvline(sdf_out['t_start_detected'] / 1e-9, ls='--', c='r')
    ax3.axvline(sdf_out['t_start_corrected'] / 1e-9, ls='-', c='r')
    if inputs['start_time_user'] == 'none':
        ax3.axhline(sdf_out['f_doi'][sdf_out['f_doi_carr_top_idx']] / 1e9, c='r')
    ax3.set_ylim([inputs['freq_min'] / 1e9, inputs['freq_max'] / 1e9])
    ax3.set_xlim([sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9])
    plt3.set_clim([np.min(sdf_out['power_doi']), np.max(sdf_out['power_doi'])])
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Frequency (GHz)')
    ax3.set_title('ROI')
    ax3.minorticks_on()

    # plotting the filtered spectrogram ROI
    plt4 = ax4.imshow(cf_out['power_filt'], aspect='auto', origin='lower',
                      interpolation='none',
                      extent=[cf_out['t_filt'][0] / 1e-9, cf_out['t_filt'][-1] / 1e-9,
                              cf_out['f_filt'][0] / 1e9, cf_out['f_filt'][-1] / 1e9],
                      cmap=inputs['cmap'])
    fig.colorbar(plt4, ax=ax4, label='Power (dB)')
    ax4.axvline(sdf_out['t_start_detected'] / 1e-9, ls='--', c='r')
    ax4.axvline(sdf_out['t_start_corrected'] / 1e-9, ls='-', c='r')
    ax4.set_ylim([inputs['freq_min'] / 1e9, inputs['freq_max'] / 1e9])
    ax4.set_xlim([sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9])
    plt4.set_clim([np.min(cf_out['power_filt_doi']), np.max(cf_out['power_filt_doi'])])
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Frequency (GHz)')
    ax4.set_title('Filtered')
    ax4.minorticks_on()

    # plotting the velocity and smoothed velocity curves to be overlaid on top of the spectrogram
    ax5.plot((vc_out['time_f']) / 1e-9,
             vc_out['velocity_f'], '-', c='grey', alpha=0.65, linewidth=3.5, label='Velocity')
    ax5.plot((vc_out['time_f']) / 1e-9,
             vc_out['velocity_f_smooth'], 'k-', linewidth=2, label='Smoothed Velocity')
    ax5.set_xlabel('Time (ns)')
    ax5.set_ylabel('Velocity (m/s)')
    ax5.legend(loc='lower right', fontsize=9, framealpha=1)
    ax5.set_title('Velocity vs Frequency')
    ax5.set_zorder(1)
    ax5.patch.set_visible(False)

    # plotting the final spectrogram to go with the velocity curves
    plt6 = ax6.imshow(cf_out['power_filt'],
                      extent=[cf_out['t_filt'][0] / 1e-9, cf_out['t_filt'][-1] / 1e-9,
                              cf_out['f_filt'][0] / 1e9, cf_out['f_filt'][-1] / 1e9],
                      aspect='auto',
                      origin='lower',
                      interpolation='none',
                      cmap=inputs['cmap'])
    ax6.set_ylabel('Frequency (GHz)')
    vel_lim = np.array([-300, np.max(vc_out['velocity_f_smooth']) + 300])
    ax5.set_ylim(vel_lim)
    ax5.set_xlim([cf_out['t_filt'][0] / 1e-9, cf_out['t_filt'][-1] / 1e-9])
    freq_lim = (vel_lim / (inputs['lam'] / 2)) + cen
    ax6.set_ylim(freq_lim / 1e9)
    ax6.set_xlim([sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9])
    ax6.minorticks_on()
    plt6.set_clim([np.min(cf_out['power_filt_doi']), np.max(cf_out['power_filt_doi'])])

    # plotting the final smoothed velocity trace with spall point markers (if they were found on the signal)
    ax7.plot((vc_out['time_f'] - sdf_out['t_start_corrected']) / 1e-9,
             vc_out['velocity_f_smooth'], 'k-', linewidth=2.5)
    ax7.set_xlabel('Time (ns)')
    ax7.set_ylabel('Velocity (m/s)')
    if not np.isnan(sa_out['t_max_comp']):
        ax7.plot((sa_out['t_max_comp'] - sdf_out['t_start_corrected']) / 1e-9, sa_out['v_max_comp'], 'bs',
                 label=f'Velocity at Max Compression: {int(round(sa_out["v_max_comp"]))}')
    if not np.isnan(sa_out['t_max_ten']):
        ax7.plot((sa_out['t_max_ten'] - sdf_out['t_start_corrected']) / 1e-9, sa_out['v_max_ten'], 'ro',
                 label=f'Velocity at Max Tension: {int(round(sa_out["v_max_ten"]))}')
    if not np.isnan(sa_out['t_rc']):
        ax7.plot((sa_out['t_rc'] - sdf_out['t_start_corrected']) / 1e-9, sa_out['v_rc'], 'gD',
                 label=f'Velocity at Recompression: {int(round(sa_out["v_rc"]))}')
    ax7.set_xlim([-inputs['t_before'] / 1e-9, (vc_out['time_f'][-1] - sdf_out['t_start_corrected']) / 1e-9])
    ax7.set_title('Free Surface Velocity')
    if not np.isnan(sa_out['t_max_comp']) or not np.isnan(sa_out['t_max_ten']) or not np.isnan(sa_out['t_rc']):
        ax7.legend(loc='lower right', fontsize=9)

    # table 1 to show general information on the run
    run_data1 = {'Name': ['Date',
                          'Time',
                          'File Name',
                          'Run Time'],
                 'Value': [start_time.strftime('%b %d %Y'),
                           start_time.strftime('%I:%M %p'),
                           inputs['filename'],
                           (end_time - start_time)]}

    df1 = pd.DataFrame(data=run_data1)
    cellLoc1 = 'center'
    loc1 = 'center'
    table1 = ax8.table(cellText=df1.values,
                       colLabels=df1.columns,
                       cellLoc=cellLoc1,
                       loc=loc1)
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.1, 2.25)
    ax8.axis('tight')
    ax8.axis('off')

    # table 2 to show important results from the run
    run_data2 = {'Name': ['Velocity at Max Compression (m/s)',
                          'Peak Shock Stress (GPa)',
                          'Strain Rate (x1e6)',
                          'Spall Strength (GPa)'],
                 'Value': [round(sa_out['v_max_comp'], 2),
                           round((.5 * inputs['density'] * inputs['C0'] * sa_out['v_max_comp']) / 1e9, 6),
                           f"{round(sa_out['strain_rate_est'] / 1e6, 6)} +- {round(fua_out['strain_rate_uncert'] / 1e6, 6)}",
                           f"{round(sa_out['spall_strength_est'] / 1e9, 6)} +- {round(fua_out['spall_uncert'] / 1e9, 6)}"]}

    df2 = pd.DataFrame(data=run_data2)
    cellLoc2 = 'center'
    loc2 = 'center'
    table2 = ax9.table(cellText=df2.values,
                       colLabels=df2.columns,
                       cellLoc=cellLoc2,
                       loc=loc2)
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.1, 2.25)
    ax9.axis('tight')
    ax9.axis('off')

    # fix the layout
    plt.tight_layout()

    # display the plots if desired. if this is turned off the plots will still save
    if inputs['display_plots'] == 'yes':
        plt.show()

    # return the figure so it can be saved if desired
    return fig
