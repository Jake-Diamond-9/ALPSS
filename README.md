# ALPSS: A program for the automated analysis of photonic Doppler velocimetry spall signals
## Copyright
GNU General Public License v3.0

## Overview
ALPSS (<b><i>A</i></b>&#8202;na<b><i>L</i></b>&#8202;ysis of <b><i>P</i></b>&#8202;hotonic Doppler velocimetry <b><i>S</i></b>&#8202;pall <b><i>S</i></b>&#8202;ignals) was developed to automate the processing of PDV spall signals. This readme is a simple quick-start guide. For comprehensive documentation please refer to the repository wiki. Any suggestions or bugs can be reported to <jdiamo15@jhu.edu>.

## Installation
It is recommended for users new to python to use [Anaconda](https://www.anaconda.com/).

### Installing Required Packages
ALPSS requires the following packages:
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [opencv](https://docs.opencv.org/4.x/d7/dbd/group__imgproc.html)
- [watchdog](https://pythonhosted.org/watchdog/)

The simplest way to install these packages is to use the Anaconda interface.
1. Open Anaconda and navigate to the **_Environments_** tab on the left hand side underneath **_Home_**.
2. Towards the bottom of the window click **_Create_** to create a new environment.
3. Name your new environment ("ALPSS" is recommended) and use the dropdown menu to install the latest version of Python.
4. In the ALPSS environment click the dropdown menu that says **_Installed_** and change it to **_All_**.
5. In the **_Search Packages_** bar type "matplotlib". Select the box next to the package with the name **_matplotlib_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.
6. In the **_Search Packages_** bar type "numpy". Select the box next to the package with the name **_numpy_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.
7. In the **_Search Packages_** bar type "scipy". Select the box next to the package with the name **_scipy_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.
8. In the **_Search Packages_** bar type "pandas". Select the box next to the package with the name **_pandas_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.
9. In the **_Search Packages_** bar type "opencv". Select the 3 boxes next to the packages with the names **_libopencv, opencv, and, py-opencv_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.
10. In the **_Search Packages_** bar type "watchdog". Select the box next to the package with the name **_watchdog_** and click the green **_Apply_** button in the bottom right. In the new package install window click the green **_Apply_** button.

### Downloading ALPSS Code
The ALPSS program files can be downloaded from the main page of the repository. Go to the green **_Code_** dropdown menu and select **_Download ZIP_** to download all files in the main branch of the repository. Then move the files to your desired directory.

## Running ALPSS
In your ALPSS python environment return to the Anaconda home screen. Locate the **_Spyder_** application and click **_Install_**. Once complete, click **_Launch_** to open Spyder. Next click _File_ -> _Open_, navigate to the directory with the ALPSS program files, and open the file _alpss\_run.py_.

### Running a Single Signal

### Running Signals with Automatic File Detection

## Example Usage
