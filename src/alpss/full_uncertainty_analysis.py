"""
Based on the work of Mallick et al.

Mallick, D.D., Zhao, M., Parker, J. et al. Laser-Driven Flyers and Nanosecond-Resolved Velocimetry for Spall Studies
in Thin Metal Foils. Exp Mech 59, 611–628 (2019). https://doi.org/10.1007/s11340-019-00519-x
"""

import numpy as np


# program to calculate the uncertainty in the spall strength and strain rate
def full_uncertainty_analysis(cen, sa_out, iua_out, **inputs):
    """
    Based on the work of Mallick et al.

    Mallick, D.D., Zhao, M., Parker, J. et al. Laser-Driven Flyers and Nanosecond-Resolved Velocimetry for Spall Studies
    in Thin Metal Foils. Exp Mech 59, 611–628 (2019). https://doi.org/10.1007/s11340-019-00519-x
    """

    # unpack dictionary values in to individual variables
    rho = inputs["density"]
    C0 = inputs["C0"]
    lam = inputs["lam"]
    delta_rho = inputs["delta_rho"]
    delta_C0 = inputs["delta_C0"]
    delta_lam = inputs["delta_lam"]
    theta = inputs["theta"]
    delta_theta = inputs["delta_theta"]
    delta_freq_tb = sa_out["peak_velocity_freq_uncert"]
    delta_freq_td = sa_out["max_ten_freq_uncert"]
    delta_time_c = iua_out["tau"]
    delta_time_d = iua_out["tau"]
    freq_tb = (sa_out["v_max_comp"] * 2) / lam + cen
    freq_td = (sa_out["v_max_ten"] * 2) / lam + cen
    time_c = sa_out["t_max_comp"]
    time_d = sa_out["t_max_ten"]

    # assuming time c is the same as time b
    freq_tc = freq_tb
    delta_freq_tc = delta_freq_tb

    # convert angles to radians
    theta = theta * (np.pi / 180)
    delta_theta = delta_theta * (np.pi / 180)

    # calculate the individual terms for spall uncertainty
    term1 = (
        -0.5
        * rho
        * C0
        * (lam / 2)
        * np.tan(theta)
        * (1 / np.cos(theta))
        * (freq_tb - freq_td)
        * delta_theta
    )
    term2 = 0.5 * rho * C0 * (lam / (2 * np.cos(theta))) * delta_freq_tb
    term3 = -0.5 * rho * C0 * (lam / (2 * np.cos(theta))) * delta_freq_td
    term4 = 0.5 * rho * C0 * (1 / (2 * np.cos(theta))) * (freq_tb - freq_td) * delta_lam
    term5 = 0.5 * rho * (lam / (2 * np.cos(theta))) * (freq_tb - freq_td) * delta_C0
    term6 = 0.5 * C0 * (lam / (2 * np.cos(theta))) * (freq_tb - freq_td) * delta_rho

    # calculate spall uncertainty
    delta_spall = np.sqrt(
        term1**2 + term2**2 + term3**2 + term4**2 + term5**2 + term6**2
    )

    # calculate the individual terms for strain rate uncertainty
    d_f = freq_tc - freq_td
    d_t = time_d - time_c
    term7 = (-lam / (4 * C0**2 * np.cos(theta))) * (d_f / d_t) * delta_C0
    term8 = (1 / (4 * C0 * np.cos(theta))) * (d_f / d_t) * delta_lam
    term9 = (
        ((lam * np.tan(theta)) / (4 * C0 * np.cos(theta))) * (d_f / d_t) * delta_theta
    )
    term10 = (lam / (4 * C0 * np.cos(theta))) * (1 / d_t) * delta_freq_tc
    term11 = (-lam / (4 * C0 * np.cos(theta))) * (1 / d_t) * delta_freq_td
    term12 = (-lam / (4 * C0 * np.cos(theta))) * (d_f / d_t**2) * delta_time_c
    term13 = (lam / (4 * C0 * np.cos(theta))) * (d_f / d_t**2) * delta_time_d

    # calculate strain rate uncertainty
    delta_strain_rate = np.sqrt(
        term7**2
        + term8**2
        + term9**2
        + term10**2
        + term11**2
        + term12**2
        + term13**2
    )

    # save outputs to a dictionary
    fua_out = {"spall_uncert": delta_spall, "strain_rate_uncert": delta_strain_rate}

    return fua_out