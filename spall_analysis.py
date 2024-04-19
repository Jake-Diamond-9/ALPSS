import numpy as np
from scipy import signal
import traceback


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
            max_ten_idx = extrema_min[np.where(extrema_min == np.argmax(velocity_f_smooth))[0][0] + 1 + pb_idx_correction]



            max_ten_freq_uncert = freq_uncert[max_ten_idx]
            max_ten_vel_uncert = vel_uncert[max_ten_idx]



            max_tension_velocity = velocity_f_smooth[max_ten_idx]









            pullback_velocity = peak_velocity - max_tension_velocity

            # calculate the estimated strain rate and spall strength
            strain_rate_est = (0.5 / C0) * pullback_velocity / (time_f[max_ten_idx] - time_f[np.argmax(velocity_f_smooth)])
            spall_strength_est = 0.5 * density * C0 * pullback_velocity

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
