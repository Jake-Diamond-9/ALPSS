import numpy as np


# define function for a normal distribution
def gauss(x, amp, sigma, mu):
    f = (amp / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return f


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
