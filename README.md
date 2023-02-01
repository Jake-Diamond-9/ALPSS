# ALPSS: A program for the automated analysis of photonic Doppler velocimetry spall signals
## Copyright
GNU General Public License v3.0

## Overview
ALPSS (<b><i>A</i></b>&#8202;na<b><i>L</i></b>&#8202;ysis of <b><i>P</i></b>&#8202;hotonic Doppler velocimetry <b><i>S</i></b>&#8202;pall <b><i>S</i></b>&#8202;ignals) was developed to automate the processing of PDV spall signals. This readme is a simple quick-start guide. For comprehensive documentation please refer to the repository wiki. Any suggestions or bugs can be reported to <jdiamo15@jhu.edu>.

## Citing
For use in published works, ALPSS can be citied from its original paper _ALPSS: A program for automated analysis of photonic Doppler velocimetry spall signals, in prep._

## Installation
It is recommended for users new to python to use [Anaconda](https://www.anaconda.com/).

### Downloading ALPSS Code
The ALPSS program files can be downloaded from the main page of the repository. Go to the green **_Code_** dropdown menu and select **_Download ZIP_** to download all files from the main branch of the repository. Then move the files to your desired directory.

### Installing Required Packages
ALPSS requires the following packages:
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [opencv](https://docs.opencv.org/4.x/d7/dbd/group__imgproc.html)
- [watchdog](https://pythonhosted.org/watchdog/)

The simplest way to install these packages is to use the prepared Anaconda environment in the file _ALPSS\_env.yml_. To do this:
1. Open Anaconda and navigate to the **_Environments_** tab on the left hand side underneath **_Home_**.
2. Towards the bottom of the window click **_Import_**.
3. From your local drive, select the file _ALPSS\_env.yml_. Then click the green **_Import_** button. Note this process may take a few minutes as all of the required packages are being downloaded and imported.

## Running ALPSS
In your ALPSS python environment return to the Anaconda home screen. Locate the **_Spyder_** application and click **_Install_**. Once complete, click **_Launch_** to open Spyder. Next click _File_ -> _Open_, navigate to the directory with the ALPSS program files, and open the file _alpss\_run.py_ to run a single signal or _alpss\_auto\_run.py_ to use automatic file detection.

### Running a Single Signal

### Running Signals with Automatic File Detection
