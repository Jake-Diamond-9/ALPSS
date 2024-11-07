"""A DataFileStreamHandler that triggers some arbitrary local code when full files are available"""

from abc import ABC, abstractmethod
import datetime
from openmsistream.data_file_io.actor.data_file_stream_processor import (
    DataFileStreamProcessor,
)

from openmsistream.utilities.config import RUN_CONST

import re
import sys
from io import BytesIO

from ALPSS.alpss_main import alpss_main


class ALPSStreamProcessor(DataFileStreamProcessor):
    """
    A class to consume :class:`~.data_file_io.entity.data_file_chunk.DataFileChunk` messages
    into memory and perform some operation(s) when entire files are available.
    This is a base class that cannot be instantiated on its own.

    :param config_path: Path to the config file to use in defining the Broker connection
        and Consumers
    :type config_path: :class:`pathlib.Path`
    :param topic_name: Name of the topic to which the Consumers should be subscribed
    :type topic_name: str
    :param output_dir: Path to the directory where the log and csv registry files should be kept
        (if None a default will be created in the current directory)
    :type output_dir: :class:`pathlib.Path`, optional
    :param mode: a string flag determining whether reconstructed data files should
        have their contents stored only in "memory" (the default, and the fastest),
        only on "disk" (in the output directory, to reduce the memory footprint),
        or "both" (for flexibility in processing)
    :type mode: str, optional
    :param datafile_type: the type of data file that recognized files should be reconstructed as.
        Default options are set automatically depending on the "mode" argument.
        (must be a subclass of :class:`~.data_file_io.DownloadDataFile`)
    :type datafile_type: :class:`~.data_file_io.DownloadDataFile`, optional
    :param n_threads: the number of threads/consumers to run
    :type n_threads: int, optional
    :param consumer_group_id: the group ID under which each consumer should be created
    :type consumer_group_id: str, optional
    :param filepath_regex: If given, only messages associated with files whose paths match
        this regex will be consumed
    :type filepath_regex: :type filepath_regex: :func:`re.compile` or None, optional

    :raises ValueError: if `datafile_type` is not a subclass of
        :class:`~.data_file_io.DownloadDataFileToMemory`, or more specific as determined
        by the "mode" argument
    """

    def __init__(
        self,
        config_file,
        topic_name,
        out_files_dir,
        download_regex,
        **kwargs,
    ):

        super().__init__(config_file, topic_name, **kwargs)
        self.out_files_dir = out_files_dir
        self.output_dir = kwargs['output_dir']
        # either create an engine to interact with a DB, or store the path to the output file
        self._engine = None
        self._output_file = None

    def _process_downloaded_data_file(self, datafile, lock):
        """
        Perform some arbitrary operation(s) on a given data file that has been fully read
        from the stream. Can optionally lock other threads using the given lock.

        Not implemented in the base class.

        :param datafile: A :class:`~.data_file_io.DownloadDataFileToMemory` object that
            has received all of its messages from the topic
        :type datafile: :class:`~.data_file_io.DownloadDataFileToMemory`
        :param lock: Acquiring this :class:`threading.Lock` object would ensure that
            only one instance of :func:`~_process_downloaded_data_file` is running at once
        :type lock: :class:`threading.Lock`

        :return: None if processing was successful, an Exception otherwise
        """

        print(f"Processing {datafile.filename}...")
        if len(datafile.bytestring) > 1000000000: # > 1GBs
            print(f"File {datafile.filename} is skipped due to large size ({len(datafile.bytestring)} bytes). ")
            return
        alpss_main(
            filename=datafile.filename,
            bytestring_data=BytesIO(datafile.bytestring),
            save_data="yes",
            start_time_user="none",
            header_lines=0,
            time_to_skip=0e-6,
            time_to_take=10e-6,
            t_before=10e-9,
            t_after=200e-9,
            start_time_correction=0e-9,
            freq_min=1e9,
            freq_max=5e9,
            smoothing_window=1001,
            smoothing_wid=3,
            smoothing_amp=1,
            smoothing_sigma=1,
            smoothing_mu=0,
            pb_neighbors=400,
            pb_idx_correction=0,
            rc_neighbors=400,
            rc_idx_correction=0,
            sample_rate=128e9,
            nperseg=512,
            noverlap=435,
            nfft=5120,
            window="hann",
            blur_kernel=(5, 5),
            blur_sigx=0,
            blur_sigy=0,
            carrier_band_time=250e-9,
            cmap="viridis",
            order=6,
            wid=15e7,
            lam=1550.016e-9,
            C0=4540,
            density=1730,
            delta_rho=9,
            delta_C0=23,
            delta_lam=8e-18,
            theta=0,
            delta_theta=5,
            # delta_freq_tb=20e6,
            # delta_freq_td=20e6,
            # delta_time_c=2.5e-9,
            # delta_time_d=2.5e-9,
            exp_data_dir=self.output_dir,
            out_files_dir=self.out_files_dir,
            display_plots="yes",
            spall_calculation="yes",
            uncert_mult=100,
            plot_figsize=(30, 10),
            plot_dpi=300,
        )
        # except Exception as exc:
        #     return exc

    @classmethod
    def run_from_command_line(cls, args=None):
        """
        Run the stream-processed analysis code from the command line
        """
        # make the argument parser
        parser = cls.get_argument_parser()
        parser.add_argument("--out_files_dir", help="Output of ALPSS algorithms")
        args = parser.parse_args(args=args)
        # make the stream processor
        alpss_analysis = cls(
            args.config,
            args.topic_name,
            args.out_files_dir,
            args.download_regex,
            output_dir=args.output_dir,
            mode=args.mode,
            n_threads=args.n_threads,
            update_secs=args.update_seconds,
            consumer_group_id=args.consumer_group_id,
        )
        # start the processor running (returns total number of messages read, processed, and names of processed files)
        run_start = datetime.datetime.now()
        msg = (
            f"Listening to the {args.topic_name} topic for flyer image files to analyze"
        )
        alpss_analysis.logger.info(msg)
        (
            n_read,
            n_processed,
            processed_filepaths,
        ) = alpss_analysis.process_files_as_read()
        alpss_analysis.close()
        run_stop = datetime.datetime.now()
        # shut down when that function returns
        msg = "Flyer analysis stream processor "
        if args.output_dir is not None:
            msg += f"writing to {args.output_dir} "
        msg += "shut down"
        alpss_analysis.logger.info(msg)
        msg = f"{n_read} total messages were consumed"
        if len(processed_filepaths) > 0:
            msg += f", {n_processed} messages were successfully processed,"
            msg += f" and the following {len(processed_filepaths)} file"
            msg += " " if len(processed_filepaths) == 1 else "s "
            msg += f"had analysis results added to {args.db_connection_str}"
        else:
            msg += f" and {n_processed} messages were successfully processed"
        msg += f" from {run_start} to {run_stop}"
        for fn in processed_filepaths:
            msg += f"\n\t{fn}"
        alpss_analysis.logger.info(msg)


def main(args=None):
    ALPSStreamProcessor.run_from_command_line(args=args)


if __name__ == "__main__":
    main()