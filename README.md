## <div align="center">**README STILL IN PROGRESS**</div>

# <div align="center">ALPSS: A program for the automated analysis of photonic Doppler velocimetry spall signals</div>
#### <div align="center">***v1.2.4***</div>

#### <div align="center">Jacob M. Diamond<sup>1,2*</sup>, K. T. Ramesh<sup>1,2</sup></div>
<div align="center"><sup>1</sup> Department of Mechanical Engineering, Johns Hopkins University, Baltimore, MD, USA </div>
<div align="center"><sup>2</sup> Hopkins Extreme Materials Institute (HEMI), Johns Hopkins University, Baltimore, MD, USA </div>
<div align="center"><sup>*</sup> jdiamo15@jhu.edu</div>
 <br>
 
<div align="center">

[![DOI](https://zenodo.org/badge/592923543.svg)](https://zenodo.org/badge/latestdoi/592923543) ![GitHub](https://img.shields.io/github/license/Jake-Diamond-9/ALPSS?color=green) ![GitHub Release Date](https://img.shields.io/github/release-date/Jake-Diamond-9/ALPSS?color=red) ![GitHub](https://img.shields.io/github/repo-size/Jake-Diamond-9/ALPSS?color=yellow)

</div>

## Overview
ALPSS (<b><i>A</i></b>&#8202;na<b><i>L</i></b>&#8202;ysis of <b><i>P</i></b>&#8202;hotonic Doppler velocimetry <b><i>S</i></b>&#8202;ignals of <b><i>S</i></b>&#8202;pall) was developed to automate the processing of PDV spall signals. This readme is a simple quick-start guide. For comprehensive documentation please refer to the repository [wiki](https://github.com/Jake-Diamond-9/ALPSS/wiki). Note that the wiki has not yet been updated for v1.2 and is still based on v1.1, although the program functionalities are largely the same. Any questions, suggestions, or bugs can be reported to <jdiamo15@jhu.edu>.

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
For users that are familiar with python you can simply clone the repo, create a virtual environment, and install the requirements in the file _requirements.txt_.

For users who are not familiar with Python, you can follow the steps below.

### Getting Started
1. If you do not already have Python installed, begin by installing [Miniconda](https://docs.anaconda.com/free/miniconda/index.html).

2. Install [VS Code](https://code.visualstudio.com/).

3. Install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extensions in VS Code. Installation instructions can be found [here](https://code.visualstudio.com/docs/editor/extension-marketplace).

4. Clone the ALPSS repo to the directory of your choice using the link <https://github.com/Jake-Diamond-9/ALPSS.git>. Instructions on cloning a repo can be found [here](https://code.visualstudio.com/docs/sourcecontrol/intro-to-git).

5. Create a virtual environment and install the packages in _requirements.txt_ by copying the following line into the terminal. Instructions on creating a virtual environment can be found [here](https://code.visualstudio.com/docs/python/environments).

~~~
pip install -r requirements.txt
~~~

ADD VIDEO TUTORIAL?

## Running ALPSS

### Running a Single Signal
Open the file _alpss\_run.py_. In the file there is a docstring that describes the input variables followed by the function **_alpss_main_**. No input parameters need to be changed from the original repository file to run the demo. The program will run the example file in the _input_data_ folder.

In the _alpss\_run_ file there is a section that reads

~~~
# %%
from alpss_main import *
import os
~~~

Just above these lines there should be small font options that read "Run Cell | Run Below | Debug Cell". Click the "Run Cell" button and the program will execute in an interactive notebook window.

Additional example data files are available through the paper by [DiMarco et al.](https://doi.org/10.3390/met13030454) and can be accessed [here](https://craedl.org/pubs?p=6348&t=3&c=187&s=hemi&d=https:%2F%2Ffs.craedl.org#publications).

Instructions on how to run your own data can be found in the repository [wiki](https://github.com/Jake-Diamond-9/ALPSS/wiki).


### Running a Signal with Automatic File Detection
SECTION ON RUNNING SIGNALS WITH _alpss_auto_run.py_

## Copyright
GNU General Public License v3.0

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
