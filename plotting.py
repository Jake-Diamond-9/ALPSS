import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import pandas as pd


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

    # noise distribution histogram
    ax2.hist(iua_out['noise'] * 1e3, bins=50, rwidth=0.8)
    ax2.set_xlabel('Noise (mV)')
    ax2.set_ylabel('Counts')

    # imported voltage spectrogram and a rectangle to show the ROI
    plt3 = ax3.imshow(10 * np.log10(sdf_out['mag'] ** 2), aspect='auto', origin='lower',
                      interpolation='none',
                      extent=[sdf_out['t'][0] / 1e-9, sdf_out['t'][-1] / 1e-9,
                              sdf_out['f'][0] / 1e9, sdf_out['f'][-1] / 1e9],
                      cmap=inputs['cmap'])
    fig.colorbar(plt3, ax=ax3, label='Power (dB)')
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

    # plotting the spectrogram of the ROI with the start-time line to see how well it lines up
    plt5 = ax5.imshow(10 * np.log10(sdf_out['mag'] ** 2), aspect='auto', origin='lower',
                      interpolation='none',
                      extent=[sdf_out['t'][0] / 1e-9, sdf_out['t'][-1] / 1e-9,
                              sdf_out['f'][0] / 1e9, sdf_out['f'][-1] / 1e9],
                      cmap=inputs['cmap'])
    fig.colorbar(plt5, ax=ax5, label='Power (dB)')
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

    # plotting the filtered spectrogram of the ROI
    plt6 = ax6.imshow(cf_out['power_filt'], aspect='auto', origin='lower',
                      interpolation='none',
                      extent=[cf_out['t_filt'][0] / 1e-9, cf_out['t_filt'][-1] / 1e-9,
                              cf_out['f_filt'][0] / 1e9, cf_out['f_filt'][-1] / 1e9],
                      cmap=inputs['cmap'])
    fig.colorbar(plt6, ax=ax6, label='Power (dB)')
    ax6.axvline(sdf_out['t_start_detected'] / 1e-9, ls='--', c='r')
    ax6.axvline(sdf_out['t_start_corrected'] / 1e-9, ls='-', c='r')
    ax6.set_ylim([inputs['freq_min'] / 1e9, inputs['freq_max'] / 1e9])
    ax6.set_xlim([sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9])
    plt6.set_clim([np.min(cf_out['power_filt_doi']), np.max(cf_out['power_filt_doi'])])
    ax6.set_xlabel('Time (ns)')
    ax6.set_ylabel('Frequency (GHz)')
    ax6.minorticks_on()

    # voltage in the ROI and the signal envelope
    ax7.plot(sdf_out['time'] / 1e-9, np.real(vc_out['voltage_filt']) * 1e3, label='Filtered Signal', c='tab:blue')
    ax7.plot(vc_out['time_f'] / 1e-9, iua_out['env_max_interp'] * 1e3, label='Signal Envelope', c='tab:red')
    ax7.plot(vc_out['time_f'] / 1e-9, iua_out['env_min_interp'] * 1e3, c='tab:red')
    ax7.set_xlabel('Time (ns)')
    ax7.set_ylabel('Voltage (mV)')
    ax7.set_xlim([sdf_out['t_doi_start'] / 1e-9, sdf_out['t_doi_end'] / 1e-9])
    ax7.legend(loc='upper right')

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
