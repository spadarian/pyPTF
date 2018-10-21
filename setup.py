#! /usr/bin/env python
"""Pedotransfer function development using Genetic Programming"""

from setuptools import setup, find_packages
import pyPTF

DESCRIPTION = __doc__
VERSION = pyPTF.__version__

setup(name='pyPTF',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      keywords="soil modelling uncertainty",
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Education',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   # 'Operating System :: Microsoft :: Windows',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   ],
      author='José Padarian',
      author_email='spadarian@gmail.com',
      url='https://github.com/spadarian/pyPTF',
      license='GPL-3.0',
      packages=find_packages(),
      zip_safe=False,
      package_data={'': ['LICENSE'],
                    'pyPTF': ['test/*.py', 'datasets/data/*']},
      install_requires=['gplearn==0.3.0',
                        'sympy==1.2',
                        'pygmo>=2.8',
                        'pandas>=0.22',
                        'matplotlib==2.2.3',
                        ],
      tests_require=['pytest'],
      setup_requires=['pytest-runner'],
      extras_require={'docs': ['Sphinx'],
                      })
