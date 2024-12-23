import unittest
from alpss.alpss_main import alpss_main
import os


class test_inputs(unittest.TestCase):
    def test_valid_time_values(self):
        with self.assertRaises(ValueError):
            alpss_main(
                filename="filename.csv",
                save_data="yes",
                start_time_user="none",
                header_lines=1,
                time_to_skip=2e-6,  # time_to_skip can't equal time_to_take
                time_to_take=2e-6,
                t_before=10e-9,
                t_after=100e-5,
                start_time_correction=0e-9,
                freq_min=1.5e9,
                freq_max=4e9,
                smoothing_window=401,
                smoothing_wid=3,
                smoothing_amp=1,
                smoothing_sigma=1,
                smoothing_mu=0,
                pb_neighbors=400,
                pb_idx_correction=0,
                rc_neighbors=400,
                rc_idx_correction=0,
                sample_rate=80e9,
                nperseg=512,
                noverlap=435,
                nfft=5120,
                window="hann",
                blur_kernel=(5, 5),
                blur_sigx=0,
                blur_sigy=0,
                carrier_band_time=250e-9,
                cmap="viridis",
                uncert_mult=100,
                order=6,
                wid=5e7,
                lam=1547.461e-9,
                C0=4540,
                density=1730,
                delta_rho=9,
                delta_C0=23,
                delta_lam=8e-18,
                theta=0,
                delta_theta=5,
                exp_data_dir=(os.getcwd() + "/input_data"),
                out_files_dir=(os.getcwd() + "/output_data"),
                display_plots="yes",
                spall_calculation="yes",
                plot_figsize=(30, 10),
                plot_dpi=300,
            )
