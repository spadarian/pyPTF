#! /usr/bin/env python
"""Pedotransfer function development using Genetic Programming"""

from setuptools import setup, find_packages
import pyPTF

DESCRIPTION = __doc__
VERSION = pyPTF.__version__

setup(name='pyPTF',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open("README.rst").read(),
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
      license='new BSD',
      packages=find_packages(),
      zip_safe=False,
      package_data={'': ['LICENSE'],
                    'pyPTF': ['test/*.py']},
      install_requires=['gplearn==0.3.0',
                        'sympy==1.2',
                        'pygmo==2.9',
                        'pandas==0.23.4',
                        'matplotlib==3.0.0',
                        ],
      tests_require=['pytest'],
      setup_requires=['pytest-runner'],
      extras_require={'docs': ['Sphinx'],
                      })
