import scipy.integrate as integrate
import numpy as np


def velocity_analysis(vc_out, **inputs):
    t_f = vc_out['time_f']
    vel_f_smooth = vc_out['velocity_f_smooth']
    spacer_thickness = inputs['spacer_thickness']
    wind = inputs['impact_vel_averaging_dist']

    # find where the velocity first goes positive. this is to cut out the large
    # artificial negative velocity at the beginning
    for i, v in enumerate(vel_f_smooth > 0):
        if v == True:
            v_pos_idx = i
            break
        else:
            continue

    # only use the data after v_pos_idx
    t = t_f[v_pos_idx:]
    # t -= t[0]
    v = vel_f_smooth[v_pos_idx:]

    # get position by trapezoidal integration of velocity
    position = (integrate.cumulative_trapezoid(v, t))

    # generate area of velocity to average over
    pos_left = spacer_thickness - wind / 2
    pos_right = spacer_thickness + wind / 2
    pos_left_idx = np.argmin(np.abs(position - pos_left))
    pos_right_idx = np.argmin(np.abs(position - pos_right))

    # calculate impact velocity as an average. skip the first entry because the
    # velocity array is 1 longer than the position array due to the integration
    impact_vel = np.mean(v[1:][pos_left_idx:pos_right_idx])
    impact_vel_SD = np.std(v[1:][pos_left_idx:pos_right_idx])

    va_out = {
        'impact_vel': impact_vel,
        'impact_vel_SD': impact_vel_SD,
        'position': position,
        'vel_position': v
    }

    return va_out
