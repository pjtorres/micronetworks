from __future__ import division

from setuptools import setup, find_packages
from glob import glob
import ast
import re

__author__ = "Pedro J. Torres"
__credits__ = ["Pedro J. Torres"]
__version__ = "0.1.0"
__email__ = "pjtorres88@gmail.com"

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('micronetworks/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(name='micronetworks',
      version='0.1.0',
      description='Microbiome/multiomics statistical analysis and network building tools.',
      classifiers=[
        'Development Status :: 0.1 - Development',
        'Programming Language :: Python :: 3.8.13',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      url='https://github.com/persephonebiome/micronetworks',
      author=__author__,
      author_email=__email__,
      packages=['micronetworks'],
      scripts=glob('micronetworks/*py'),
      install_requires=['numpy',
                        'pandas',
                        'matplotlib',
                        'seaborn',
                        'networkx',
                        'scikit-bio',
                        'community',
                        'statsmodels',
                        'scipy',
        ],
      zip_safe=False)
