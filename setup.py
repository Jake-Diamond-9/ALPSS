# imports
import pathlib
import setuptools

version = "1.2.0"

long_description = "ALPSS is a program that is designed to automatically process PDV spall signals and will allow us to keep up with high-throughput experiments. ALPSS is also designed to function as a standalone spall signal processing program. All potential inputs are located within a single function, making it easy to adjust parameters and quickly assess results."
setupkwargs = dict(
    name="ALPSS",
    packages=setuptools.find_packages(include=["ALPSS*"]),
    include_package_data=True,
    version=version,
    description=(
        "Program for the automated analysis of photonic Doppler velocimetry spall signals with uncertainty. "
    ),
    long_description=long_description,
    author="Jacob M. Diamond",
    author_email="jdiamo15@jhu.edu",
    url="https://github.com/Jake-Diamond-9/ALPSS",
    # download_url=f",
    license="GNU GPLv3",
    python_requires=">=3.7,<=3.10",
    install_requires=[],
    extras_require={},
    keywords=[
        "pdv",
        "spall",
        "high throughput",
        "automated",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)

setupkwargs["extras_require"]["all"] = sum(setupkwargs["extras_require"].values(), [])

setuptools.setup(**setupkwargs)
