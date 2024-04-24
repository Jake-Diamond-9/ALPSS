# <div align="center">_**README STILL IN PROGRESS!**_</div>


# <div align="center">ALPSS: A program for the automated analysis of photonic Doppler velocimetry spall signals</div>
#### <div align="center">***v1.2***</div>

#### <div align="center">Jacob M. Diamond<sup>1,2</sup>, K. T. Ramesh<sup>1,2</sup></div>
<div align="center"><sup>1</sup> Department of Mechanical Engineering, Johns Hopkins University, Baltimore, MD, USA </div>
<div align="center"><sup>2</sup> Hopkins Extreme Materials Institute (HEMI), Johns Hopkins University, Baltimore, MD, USA </div>
 <br>
 
<div align="center">

[![DOI](https://zenodo.org/badge/592923543.svg)](https://zenodo.org/badge/latestdoi/592923543) ![GitHub](https://img.shields.io/github/license/Jake-Diamond-9/ALPSS) ![GitHub Release Date](https://img.shields.io/github/release-date/Jake-Diamond-9/ALPSS) ![GitHub](https://img.shields.io/github/repo-size/Jake-Diamond-9/ALPSS?color=yellow)

</div>

## Copyright
GNU General Public License v3.0

## Overview
ALPSS (<b><i>A</i></b>&#8202;na<b><i>L</i></b>&#8202;ysis of <b><i>P</i></b>&#8202;hotonic Doppler velocimetry <b><i>S</i></b>&#8202;ignals of <b><i>S</i></b>&#8202;pall) was developed to automate the processing of PDV spall signals. This readme is a simple quick-start guide. For comprehensive documentation please refer to the repository [wiki](https://github.com/Jake-Diamond-9/ALPSS/wiki). Note that the wiki has not yet been updated for v1.2 and is still based on v1.1, although the program functionalities are largely the same. Any suggestions or bugs can be reported to <jdiamo15@jhu.edu>.

## Example Figure
<!---
![F2--20211018--00015--plots](https://github.com/Jake-Diamond-9/ALPSS/assets/83182690/b1e10324-27a1-4415-b294-fd93b21a75ae)
-->
<p align="center">
<img src="https://github.com/Jake-Diamond-9/ALPSS/assets/83182690/b1e10324-27a1-4415-b294-fd93b21a75ae" width="600"/>
</p>

## What's new in v1.2?
Time resolved uncertainty estimates have been added in v1.2. E.g. for any given point in time on the final velocity trace, the program will output the estimated velocity uncertainty. All other functions are essentially the same. 

## Citing ALPSS
For use in published works, ALPSS can be cited from its original paper _Automated analysis of photonic Doppler velocimetry spall signals, in submission._ 

The repository itself can be cited using its DOI 10.5281/zenodo.7603823 (v1.1 only) or with the bibtex 

~~~
@software{Diamond_ALPSS_2023,
  author = {Diamond, Jacob M. and Ramesh, K.T.},
  doi = {10.5281/zenodo.7603823},
  month = {02},
  title = {{ALPSS}},
  url = {https://github.com/Jake-Diamond-9/ALPSS},
  version = {1.1},
  year = {2023}
}
~~~

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
- [findiff](https://findiff.readthedocs.io/en/latest/)

The simplest way to install these packages is to use the prepared Anaconda environment in the file _ALPSS\_env.yml_. To do this:
1. Open Anaconda and navigate to the **_Environments_** tab on the left-hand side underneath **_Home_**.
2. Towards the bottom of the window click **_Import_**.
3. From your local drive, select the file _ALPSS\_env.yml_. Then click the green **_Import_** button. Note this process may take a little while as all the required packages are being imported. Give it at least 10-15 minutes. 

Alternatively, if the _ALPSS\_env.yml_ install does not work, the packages can be installed individually via the Anaconda Navigator interface.
1. Open Anaconda and navigate to the **_Environments_** tab on the left hand side underneath **_Home_**.
2. Towards the bottom of the window click **_Create_** to create a new environment.
3. Name your new environment ("ALPSS_env" is recommended) and use the dropdown menu to install the latest version of Python.
4. In the ALPSS environment click the dropdown menu that says **_Installed_** and change it to **_All_**.
5. In the **_Search Packages_** bar type "matplotlib". Select the box next to the package with the name **_matplotlib_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.
6. In the **_Search Packages_** bar type "numpy". Select the box next to the package with the name **_numpy_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.
7. In the **_Search Packages_** bar type "scipy". Select the box next to the package with the name **_scipy_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.
8. In the **_Search Packages_** bar type "pandas". Select the box next to the package with the name **_pandas_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.
9. In the **_Search Packages_** bar type "opencv". Select the 3 boxes next to the packages with the names **_libopencv, opencv, and, py-opencv_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.
10. In the **_Search Packages_** bar type "watchdog". Select the box next to the package with the name **_watchdog_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.
11. In the **_Search Packages_** bar type "findiff". Select the box next to the package with the name **_findiff_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.

## Running ALPSS
With your ALPSS environment selected, return to the Anaconda home screen and launch the Spyder application. Once Spyder opens, click _File_ -> _Open_, navigate to the directory with the ALPSS program files, and open the file _alpss\_run.py_ to run a single signal or _alpss\_auto\_run.py_ to use automatic file detection.

A sample file is available for download from the Johns Hopkins OneDrive [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/jdiamo15_jh_edu/EqdZ-pO-VehLkZhAY-UEVKUBBaoSvjqlMYaBigH7vllgTA?e=y3yuQt). Input data should always be in the form of a two column _.txt_ or _.csv_ file.

Additional data files are available through the paper by [DiMarco et al.](https://doi.org/10.3390/met13030454) and can be accessed [here](https://craedl.org/pubs?p=6348&t=3&c=187&s=hemi&d=https:%2F%2Ffs.craedl.org#publications).

### Running a Single Signal
Open the file _alpss\_run.py_. In the file there is a docstring that describes the input variables followed by the function **_alpss_main_**. If using the sample file provided above, no input parameters need to be changed from the original repository file. The program will run the example file in the _input_data_ folder.

In the toolbar click _Run -> Run_ and ALPSS will run the sample file. At the conclusion of the run, a figure that shows the sub processes will be displayed in the **Plots** window and a results table will be displayed in the **Console** window. Output files will be saved in the _output_data_ folder. 

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
- Professor Mark Foster
- Samuel Salander

Research was sponsored by the Army Research Laboratory and was accomplished under Cooperative Agreement Number W911NF-22-2-0014. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Office or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
