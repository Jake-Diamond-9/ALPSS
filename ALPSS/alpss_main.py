from spall_doi_finder import *
from plotting import *
from carrier_frequency import *
from carrier_filter import *
from velocity_calculation import *
from spall_analysis import *
from full_uncertainty_analysis import *
from instantaneous_uncertainty_analysis import *
from saving import *
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
import pandas as pd

def validate_inputs(inputs):
    if inputs["t_after"] > inputs["time_to_take"]:
        raise ValueError("'t_after' must be less than 'time_to_take'. ")

# main function to link together all the sub-functions
def alpss_main(**inputs):
    # validate the inputs for the run
    validate_inputs(inputs)

    # attempt to run the program in full
    try:
        # begin the program timer

        start_time = datetime.now()

        # function to find the spall signal domain of interest
        sdf_out = spall_doi_finder(**inputs)

        # function to find the carrier frequency
        cen = carrier_frequency(sdf_out, **inputs)

        # function to filter out the carrier frequency after the signal has started
        cf_out = carrier_filter(sdf_out, cen, **inputs)

        # function to calculate the velocity from the filtered voltage signal
        vc_out = velocity_calculation(sdf_out, cen, cf_out, **inputs)

        # function to estimate the instantaneous uncertainty for all points in time
        iua_out = instantaneous_uncertainty_analysis(sdf_out, vc_out, cen, **inputs)

        # function to find points of interest on the velocity trace
        sa_out = spall_analysis(vc_out, iua_out, **inputs)

        # function to calculate uncertainties in the spall strength and strain rate due to external uncertainties
        fua_out = full_uncertainty_analysis(cen, sa_out, iua_out, **inputs)

        # end the program timer
        end_time = datetime.now()

        # function to generate the final figure
        fig = plotting(
            sdf_out,
            cen,
            cf_out,
            vc_out,
            sa_out,
            iua_out,
            fua_out,
            start_time,
            end_time,
            **inputs,
        )

        # function to save the output files if desired
        if inputs["save_data"] == "yes":
            saving(
                sdf_out,
                cen,
                vc_out,
                sa_out,
                iua_out,
                fua_out,
                start_time,
                end_time,
                fig,
                **inputs,
            )

        # end final timer and display full runtime
        end_time2 = datetime.now()
        print(
            f"\nFull program runtime (including plotting and saving):\n{end_time2 - start_time}\n"
        )

    # in case the program throws an error
    except Exception:
        # print the traceback for the error
        print(traceback.format_exc())

        # attempt to plot the voltage signal from the imported data
        try:
            # import the desired data. Convert the time to skip and turn into number of rows
            t_step = 1 / inputs["sample_rate"]
            rows_to_skip = (
                inputs["header_lines"] + inputs["time_to_skip"] / t_step
            )  # skip the header lines too
            nrows = inputs["time_to_take"] / t_step

            # change directory to where the data is stored
            os.chdir(inputs["exp_data_dir"])
            data = pd.read_csv(
                inputs["filename"], skiprows=int(rows_to_skip), nrows=int(nrows)
            )

            # rename the columns of the data
            data.columns = ["Time", "Ampl"]

            # put the data into numpy arrays. Zero the time data
            time = data["Time"].to_numpy()
            time = time - time[0]
            voltage = data["Ampl"].to_numpy()

            # calculate the sample rate from the experimental data
            fs = 1 / np.mean(np.diff(time))

            # calculate the short time fourier transform
            f, t, Zxx = stft(voltage, fs, **inputs)

            # calculate magnitude of Zxx
            mag = np.abs(Zxx)

            # plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, num=2, figsize=(11, 4), dpi=300)
            ax1.plot(time / 1e-9, voltage / 1e-3)
            ax1.set_xlabel("Time (ns)")
            ax1.set_ylabel("Voltage (mV)")
            ax2.imshow(
                10 * np.log10(mag**2),
                aspect="auto",
                origin="lower",
                interpolation="none",
                extent=[t[0] / 1e-9, t[-1] / 1e-9, f[0] / 1e9, f[-1] / 1e9],
                cmap=inputs["cmap"],
            )
            ax2.set_xlabel("Time (ns)")
            ax2.set_ylabel("Frequency (GHz)")
            fig.suptitle("ERROR: Program Failed", c="r", fontsize=16)

            plt.tight_layout()
            plt.show()

        # if that also fails then print the traceback and stop running the program
        except Exception:
            print(traceback.format_exc())