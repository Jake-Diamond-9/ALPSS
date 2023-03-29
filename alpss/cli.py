"""Console script for ALPSS.

ALPSS: AnaLysis of Pdv Spall Signals
Jake Diamond (2023)
Johns Hopkins University
Hopkins Extreme Materials Institute (HEMI)
Please report any bugs or comments to jdiamo15@jhu.edu
"""

import sys
import click

from .alpss_main import alpss_main


@click.command()
@click.argument(
    "filename",
    type=click.Path(exists=True),
)
@click.option(
    "--save_data",
    type=click.Choice(["yes", "no"], case_sensitive=True),
    help="whether to save output data",
    default="yes",
)
@click.option(
    "--start_time_user",
    type=float,
    help=(
        "if not provided the program will attempt to find the "
        "signal start time automatically. if float then "
        "the program will use that as the signal start time."
    ),
    default=None,
)
@click.option(
    "--header_lines",
    type=int,
    help="number of header lines to skip in the data file",
    default=5,
)
@click.option(
    "--time_to_skip",
    type=float,
    help="the amount of time to skip in the full data file before beginning to read in data",
    default=50e-6,
)
@click.option(
    "--time_to_take",
    type=float,
    help="the amount of time to take in the data file after skipping time_to_skip",
    default=2.0e-6,
)
@click.option(
    "--t_before",
    type=float,
    help="amount of time before the signal start time to include in the velocity calculation",
    default=5e-9,
)
@click.option(
    "--t_after",
    type=float,
    help="amount of time after the signal start time to include in the velocity calculation",
    default=50e-9,
)
@click.option(
    "--start_time_correction",
    default=0e-9,
    type=float,
    help="amount of time to adjust the signal start time by",
)
@click.option(
    "--freq_min",
    type=float,
    default=1.5e9,
    help="minimum frequency for the region of interest",
)
@click.option(
    "--freq_max",
    type=float,
    default=4e9,
    help="maximum frequency for the region of interest",
)
@click.option(
    "--smoothing_window",
    default=401,
    type=int,
    help="number of points to use for the smoothing window. must be an odd number",
)
@click.option(
    "--smoothing_wid",
    default=3.0,
    type=float,
    help=(
        "half the width of the normal distribution used "
        "to calculate the smoothing weights (recommend 3)"
    ),
)
@click.option(
    "--smoothing_amp",
    default=1.0,
    type=float,
    help=(
        "amplitude of the normal distribution used to calculate the smoothing weights "
        "(recommend 1)"
    ),
)
@click.option(
    "--smoothing_sigma",
    default=1.0,
    type=float,
    help=(
        "standard deviation of the normal distribution used to calculate the smoothing weights "
        "(recommend 1)"
    ),
)
@click.option(
    "--smoothing_mu",
    default=0.0,
    type=float,
    help=(
        "mean of the normal distribution used to calculate the smoothing weights (recommend 0)"
    ),
)
@click.option(
    "--pb_neighbors",
    default=400,
    type=int,
    help="number of neighbors to compare to when searching for the pullback local minimum",
)
@click.option(
    "--pb_idx_correction",
    default=0,
    type=int,
    help="number of local minima to adjust by if the program grabs the wrong one",
)
@click.option(
    "--rc_neighbors",
    default=400,
    type=int,
    help="number of neighbors to compare to when searching for the recompression local maximum",
)
@click.option(
    "--rc_idx_correction",
    default=0,
    type=int,
    help="number of local maxima to adjust by if the program grabs the wrong one",
)
@click.option(
    "--sample_rate",
    default=80e9,
    type=float,
    help="sample rate of the oscilloscope used in the experiment",
)
@click.option(
    "--nperseg",
    default=512,
    type=int,
    help="number of points to use per segment of the stft",
)
@click.option(
    "--noverlap",
    default=435,
    type=int,
    help="number of points to overlap per segment of the stft",
)
@click.option(
    "--nfft",
    default=5120,
    type=int,
    help="number of points to zero pad per segment of the stft",
)
@click.option(
    "--window",
    default="hann",
    type=str,
    help="window function to use for the stft (recommend 'hann')",  ####
)
@click.option(
    "--blur_kernel",
    default=(5, 5),
    type=click.Tuple([int, int]),
    help="kernel size for gaussian blur smoothing (recommend (5, 5))",
)
@click.option(
    "--blur_sigx",
    default=0.0,
    type=float,
    help="standard deviation of the gaussian blur kernel in the x direction (recommend 0)",
)
@click.option(
    "--blur_sigy",
    default=0.0,
    type=float,
    help="standard deviation of the gaussian blur kernel in the y direction (recommend 0)",
)
@click.option(
    "--carrier_band_time",
    default=250e-9,
    type=float,
    help=(
        "length of time from the beginning of the imported data window to average "
        "the frequency of the top of the carrier band in the thresholded spectrogram"
    ),
)
@click.option(
    "--cmap",
    default="viridis",
    type=str,
    help="colormap for the spectrograms (recommend 'viridis')",
)
@click.option(
    "--order",
    default=6,
    type=int,
    help="order for the gaussian notch filter used to remove the carrier band (recommend 6)",
)
@click.option(
    "--wid",
    default=5e7,
    type=float,
    help="width of the gaussian notch filter used to remove the carrier band (recommend 1e8)",
)
@click.option(
    "--lam",
    default=1547.461e-9,
    type=float,
    help="wavelength of the target laser",
)
@click.option(
    "--C0",
    "C0",
    default=4540.0,
    type=float,
    help="bulk wavespeed of the sample",
)
@click.option(
    "--density",
    default=1730.0,
    type=float,
    help="density of the sample",
)
@click.option(
    "--delta_rho",
    default=9.0,
    type=float,
    help="uncertainty in density of the sample",
)
@click.option(
    "--delta_C0",
    "delta_C0",
    default=23.0,
    type=float,
    help="uncertainty in the bulk wavespeed of the sample",
)
@click.option(
    "--delta_lam",
    default=8e-18,
    type=float,
    help="uncertainty in the wavelength of the target laser",
)
@click.option(
    "--theta",
    default=0.0,
    type=float,
    help="angle of incidence of the PDV probe",
)
@click.option(
    "--delta_theta",
    default=2.5,
    type=float,
    help="uncertainty in the angle of incidence of the PDV probe",
)
@click.option(
    "--delta_freq_tb",
    default=20e6,
    type=float,
    help="uncertainty in the frequency at time b",
)
@click.option(
    "--delta_freq_td",
    default=20e6,
    type=float,
    help="uncertainty in the frequency at time d",
)
@click.option(
    "--delta_time_c",
    default=2.5e-9,
    type=float,
    help="uncertainty in time c",
)
@click.option(
    "--delta_time_d",
    default=2.5e-9,
    help="uncertainty in time d",
)
@click.option(
    "--exp_data_dir",
    default=".",
    type=str,
    help="directory from which to read the experimental data file",
)
@click.option(
    "--out_files_dir",
    default="outputs",
    type=str,
    help="directory to save output data to",
)
@click.option(
    "--display_plots",
    type=click.Choice(["yes", "no"], case_sensitive=True),
    default="yes",
    help=(
        "'yes' to display the final plots and 'no' to not display them. if save_data='yes' "
        "and and display_plots='no' the plots will be saved but not displayed"
    ),
)
@click.option(
    "--plot_figsize",
    default=(12, 10),
    type=click.Tuple([float, float]),
    help="figure size for the final plots in inches",
)
@click.option("--plot_dpi", type=int, help="dpi for the final plots", default=300)
def alpss(
    filename,
    save_data,
    start_time_user,
    header_lines,
    time_to_skip,
    time_to_take,
    t_before,
    t_after,
    start_time_correction,
    freq_min,
    freq_max,
    smoothing_window,
    smoothing_wid,
    smoothing_amp,
    smoothing_sigma,
    smoothing_mu,
    pb_neighbors,
    pb_idx_correction,
    rc_neighbors,
    rc_idx_correction,
    sample_rate,
    nperseg,
    noverlap,
    nfft,
    window,
    blur_kernel,
    blur_sigx,
    blur_sigy,
    carrier_band_time,
    cmap,
    order,
    wid,
    lam,
    C0,
    density,
    delta_rho,
    delta_C0,
    delta_lam,
    theta,
    delta_theta,
    delta_freq_tb,
    delta_freq_td,
    delta_time_c,
    delta_time_d,
    exp_data_dir,
    out_files_dir,
    display_plots,
    plot_figsize,
    plot_dpi,
):
    """Run ALPSS on FILENAME."""
    if start_time_user is None:
        start_time_user = "none"

    alpss_main(
        filename=filename,
        save_data=save_data,
        start_time_user=start_time_user,
        header_lines=header_lines,
        time_to_skip=time_to_skip,
        time_to_take=time_to_take,
        t_before=t_before,
        t_after=t_after,
        start_time_correction=start_time_correction,
        freq_min=freq_min,
        freq_max=freq_max,
        smoothing_window=smoothing_window,
        smoothing_wid=smoothing_wid,
        smoothing_amp=smoothing_amp,
        smoothing_sigma=smoothing_sigma,
        smoothing_mu=smoothing_mu,
        pb_neighbors=pb_neighbors,
        pb_idx_correction=pb_idx_correction,
        rc_neighbors=rc_neighbors,
        rc_idx_correction=rc_idx_correction,
        sample_rate=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window=window,
        blur_kernel=blur_kernel,
        blur_sigx=blur_sigx,
        blur_sigy=blur_sigy,
        carrier_band_time=carrier_band_time,
        cmap=cmap,
        order=order,
        wid=wid,
        lam=lam,
        C0=C0,
        density=density,
        delta_rho=delta_rho,
        delta_C0=delta_C0,
        delta_lam=delta_lam,
        theta=theta,
        delta_theta=delta_theta,
        delta_freq_tb=delta_freq_tb,
        delta_freq_td=delta_freq_td,
        delta_time_c=delta_time_c,
        delta_time_d=delta_time_d,
        exp_data_dir=exp_data_dir,
        out_files_dir=out_files_dir,
        display_plots=display_plots,
        plot_figsize=plot_figsize,
        plot_dpi=plot_dpi,
    )

    return 0


if __name__ == "__main__":
    sys.exit(alpss())  # pragma: no cover
