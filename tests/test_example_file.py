import unittest
import os
import pandas as pd
from alpss.alpss_main import alpss_main


class TestExampleFile(unittest.TestCase):
    def setUp(self):
        # Set up paths relative to the test directory
        self.input_file = os.path.join(os.getcwd(), "input_data", "example_file.csv")

        # Load the expected inputs
        self.expected_inputs = {
            "filename": "example_file.csv",
            "save_data": "yes",
            "start_time_user": "none",
            "header_lines": 0,
            "time_to_skip": 0e-6,
            "time_to_take": 10e-6,
            "t_before": 10e-9,
            "t_after": 200e-9,
            "start_time_correction": 0e-9,
            "freq_min": 1e9,
            "freq_max": 5e9,
            "smoothing_window": 1001,
            "smoothing_wid": 3,
            "smoothing_amp": 1,
            "smoothing_sigma": 1,
            "smoothing_mu": 0,
            "pb_neighbors": 400,
            "pb_idx_correction": 0,
            "rc_neighbors": 400,
            "rc_idx_correction": 0,
            "sample_rate": 128e9,
            "nperseg": 512,
            "noverlap": 435,
            "nfft": 5120,
            "window": "hann",
            "blur_kernel": (5, 5),
            "blur_sigx": 0,
            "blur_sigy": 0,
            "carrier_band_time": 250e-9,
            "cmap": "viridis",
            "uncert_mult": 100,
            "order": 6,
            "wid": 15e7,
            "lam": 1550.016e-9,
            "C0": 4540,
            "density": 1730,
            "delta_rho": 9,
            "delta_C0": 23,
            "delta_lam": 8e-18,
            "theta": 0,
            "delta_theta": 5,
            "exp_data_dir": "/srv/hemi01-j01/ALPSS/tests/input_data",
            "out_files_dir": "/srv/hemi01-j01/ALPSS/tests/output_data2",
            "display_plots": "yes",
            "spall_calculation": "yes",
            "plot_figsize": (30, 10),
            "plot_dpi": 300,
        }

        # Define the expected output as a dictionary
        self.expected_results = {
            "Date": "Nov 14 2024",
            "Time": "12:47 PM",
            "File Name": "example_file.csv",
            "Run Time": "0:00:05.788687",
            "Velocity at Max Compression": 873.5450985331408,
            "Time at Max Compression": 3.041925250003852e-06,
            "Velocity at Max Tension": 359.74496364236944,
            "Time at Max Tension": 3.073312760004665e-06,
            "Velocity at Recompression": 364.3168301692125,
            "Time at Recompression": 3.076412760004421e-06,
            "Carrier Frequency": 2233199812.41015,
            "Spall Strength": 2017744509.7295487,
            "Spall Strength Uncertainty": 23609426.814541273,
            "Strain Rate": 1802816.4049345185,
            "Strain Rate Uncertainty": 396836.143378536,
            "Peak Shock Stress": 3430498956.449497,
            "Spect Time Res": 9.625000808504594e-10,
            "Spect Freq Res": 15624998.687492654,
            "Spect Velocity Res": 12.109498982796307,
            "Signal Start Time": 2.9154127448960416e-06,
            "Smoothing Characteristic Time": 4.879875409911828e-09,
        }

    def test_alpss_main(self):
        # Ensure the input file exists
        self.assertTrue(os.path.isfile(self.input_file), "Input file not found.")

        # Call the function with the parameters
        result = alpss_main(**self.expected_inputs)
        print(result.keys())

        # Check each expected result value individually
        for key, expected_value in self.expected_results.items():
            with self.subTest(key=key):
                self.assertIn(key, result, f"{key} not found in the result")
                self.assertAlmostEqual(
                    result[key],
                    expected_value,
                    places=5,
                    msg=f"Value for {key} does not match. Expected {expected_value}, got {result[key]}",
                )


if __name__ == "__main__":
    unittest.main()
