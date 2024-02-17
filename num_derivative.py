import numpy as np
import findiff
# https://github.com/maroba/findiff/tree/master


# function to take the numerical derivative of input array phas (central difference with a 9-point stencil).
# phas is padded so that after smoothing the final velocity trace matches the length of the domain of interest.
# this avoids issues with handling the boundaries in the derivative and later in smoothing.
def num_derivative(phas, window, time_start_idx, time_end_idx, fs):

    acc = 8

    # calculate how much padding is needed. half_space padding comes from the length of the smoothing window. the 4
    # is added to account for the 9-point stencil
    half_space = int(np.floor(window / 2))
    pad = int(half_space + acc/2)

    # get only the section of interest
    phas_pad = phas[time_start_idx - pad:time_end_idx + pad]

    ddt = findiff.FinDiff(0, 1 / fs, 1, acc=acc)
    dpdt_pad = ddt(phas_pad) * (1/(2*np.pi))



    # # calculate the derivative with the 9-point central difference method
    # dpdt_pad = np.zeros(phas_pad.shape)
    # for i in range(4, len(dpdt_pad) - 4):
    #     dpdt_pad[i] = ((1 / 280) * phas_pad[i - 4]
    #                    + (-4 / 105) * phas_pad[i - 3]
    #                    + (1 / 5) * phas_pad[i - 2]
    #                    + (-4 / 5) * phas_pad[i - 1]
    #                    + (4 / 5) * phas_pad[i + 1]
    #                    + (-1 / 5) * phas_pad[i + 2]
    #                    + (4 / 105) * phas_pad[i + 3]
    #                    + (-1 / 280) * phas_pad[i + 4]) \
    #                   * (fs / (2 * np.pi))


    # output both the padded and un-padded derivatives
    dpdt = dpdt_pad[pad:-pad]
    dpdt_pad = dpdt_pad[int(acc/2):-int(acc/2)]

    return dpdt, dpdt_pad
