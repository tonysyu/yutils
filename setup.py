#! /usr/bin/env python
from distutils.core import setup

import setuptools


with open('README.md') as f:
    long_description = f.read()

setup(name='yutils',
      version='0.1',
      author='Tony S. Yu',
      description='Utility functions',
      long_description=long_description,
      author_email='tsyu80@gmail.com',
      packages=setuptools.find_packages(),
      include_package_data=True,
     )
