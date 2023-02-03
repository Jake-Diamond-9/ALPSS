# <div align="center">ALPSS: A program for the automated analysis of photonic Doppler velocimetry spall signals</div>
#### <div align="center">***v1.1***</div>

#### <div align="center">Jacob M. Diamond<sup>1,2</sup>, Samuel Salander<sup>3</sup>, K. T. Ramesh<sup>1,2</sup></div>
<div align="center"><sup>1</sup> Department of Mechanical Engineering, Johns Hopkins University, Baltimore, MD, USA </div>
<div align="center"><sup>2</sup> Hopkins Extreme Materials Institute (HEMI), Johns Hopkins University, Baltimore, MD, USA </div>
<div align="center"><sup>3</sup> Department of Physics and Astronomy, Johns Hopkins University, Baltimore, MD, USA </div>  
 <br>
 
<div align="center">

[![DOI](https://zenodo.org/badge/592923543.svg)](https://zenodo.org/badge/latestdoi/592923543) ![GitHub](https://img.shields.io/badge/license-GPL--3.0-orange) ![PyPI - Python Version](https://img.shields.io/badge/python-3.10-brightgreen) ![GitHub Release Date](https://img.shields.io/badge/release%20date-feb%202023-yellow) ![GitHub](https://img.shields.io/github/repo-size/Jake-Diamond-9/ALPSS?color=lightgrey)

</div>

## Copyright
GNU General Public License v3.0

## Overview
ALPSS (<b><i>A</i></b>&#8202;na<b><i>L</i></b>&#8202;ysis of <b><i>P</i></b>&#8202;hotonic Doppler velocimetry <b><i>S</i></b>&#8202;pall <b><i>S</i></b>&#8202;ignals) was developed to automate the processing of PDV spall signals. This readme is a simple quick-start guide. For comprehensive documentation please refer to the repository [wiki](https://github.com/Jake-Diamond-9/ALPSS/wiki). Any suggestions or bugs can be reported to <jdiamo15@jhu.edu>.

## Citing ALPSS
For use in published works, ALPSS can be cited from its original paper _ALPSS: A program for automated analysis of photonic Doppler velocimetry spall signals, in prep._ 

The repository itself can be cited using its DOI: 10.5281/zenodo.7603823 (v1.1 only)

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
1. Open Anaconda and navigate to the **_Environments_** tab on the left-hand side underneath **_Home_**.
2. Towards the bottom of the window click **_Import_**.
3. From your local drive, select the file _ALPSS\_env.yml_. Then click the green **_Import_** button. Note this process may take a few minutes as all the required packages are being imported.

## Running ALPSS
With your ALPSS environment selected, return to the Anaconda home screen and launch the Spyder application. Once Spyder opens, click _File_ -> _Open_, navigate to the directory with the ALPSS program files, and open the file _alpss\_run.py_ to run a single signal or _alpss\_auto\_run.py_ to use automatic file detection.

A sample file is available for download from the Johns Hopkins OneDrive [Here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/jdiamo15_jh_edu/EqdZ-pO-VehLkZhAY-UEVKUBBaoSvjqlMYaBigH7vllgTA?e=y3yuQt)

### Running a Single Signal
Open the file _alpss\_run.py_. In the file there is a docstring that describes the input variables followed by the function **_alpss_main_**. If using the sample file provided above, only two input parameters need to be changed from the original repository file. 
1. _exp\_data\_dir_ should be changed to the directory where you have stored the sample file.
2. _out\_files_dir_ should be changed to the directory where you would like to save the output files.

In the toolbar click _Run -> Run_ and ALPSS will run the sample file. At the conclusion of the run, a figure that shows the sub processes will be displayed in the **Plots** window and a results table will be displayed in the **Console** window. Output files will be saved in the directory specified in _out\_files_dir_. 

### Running a Signal with Automatic File Detection
Open the file _alpss\_auto\_run.py_. If using the sample file provided above, only three changes need to be changed from the original repository file.
1. In the **Watcher** class, the variable **DIRECTORY_TO_WATCH** should be changed to the directory you would like to monitor for the file.
2. _exp\_data\_dir_ should be changed to match the directory specified in **DIRECTORY_TO_WATCH**.
3. _out\_files_dir_ should be changed to the directory where you would like to save the output files.

To begin, the file you wish to run should _not_ be located in **DIRECTORY_TO_WATCH**, but instead stored in another temporary location. Then in the toolbar click _Run -> Run_ and ALPSS will begin to monitor **DIRECTORY_TO_WATCH** for changes. Next, click and drag your file from the temporary storage location in to **DIRECTORY_TO_WATCH**. The creation of the file will be automatically detected and the analysis will be run. At the conclusion of the run, a figure that shows the sub processes will be displayed in the **Plots** window and a results table will be displayed in the **Console** window. Output files will be saved in the directory specified in _out\_files_dir_. Note, the process is the same for running multiple files. All files can be moved in to the monitored directory at the same time.

## Acknowledgements and Funding
The authors would like to acknowledge the following people for their many helpful conversations and advice:
- Dr. Chris DiMarco
- Dr. Velat Killic
- Dr. Debjoy Mallcik
- Dr. Maggie Eminizer
- Dr. David Elbert

Research was sponsored by the Army Research Laboratory and was accomplished under Cooperative Agreement Number W911NF-22-2-0014. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Office or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
