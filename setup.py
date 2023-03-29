#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "Click>=7.0",
    "matplotlib",
    "numpy",
    "scipy",
    "pandas",
    "opencv-python",
]

test_requirements = []

setup(
    author="Jacob M. Diamond",
    author_email="jdiamo15@jhu.edu",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A program for the automated analysis of photonic Doppler velocimetry spall signals",
    entry_points={
        "console_scripts": [
            "alpss=alpss.cli:alpss",
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords="ALPSS",
    name="ALPSS",
    packages=find_packages(include=["alpss", "alpss.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Jake-Diamond-9/ALPSS",
    version="0.1.0",
    zip_safe=False,
)
