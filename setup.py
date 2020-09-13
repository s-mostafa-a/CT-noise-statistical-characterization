"""
setup
Module installs the ct_characterization package
Can be run via command: python setup.py install (or develop)
Author: Mostafa Ahmadi (s.mostafa.a96@gmail.com)
Created on: Sep 13, 2020
"""

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='ct_characterization',
    version='0.1.0',
    description="characterization for signal and noise of ct scan images",
    long_description=readme,
    author='Mostafa Ahmadi',
    author_email='s.mostafa.a96@gmail.com',
    url='https://github.com/s-mostafa-a/CT-noise-statistical-characterization',
    license=license,
    packages=find_packages(),
    keywords="ct scan characterization",
)

setup(install_requires=['dicom-numpy>=0.4.0',
                        'numpy>=1.19.2',
                        'SimpleITK>=1.2.4',
                        'pydicom>=2.0.0',
                        'matplotlib>=3.3.1'], **args)
