#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3',
                     "torch>=1.0.0",
                     "FrEIA>=0.2",
                     "PyGSP @ git+https://github.com/epfl-lts2/pygsp.git@39a0665f637191152605911cf209fc16a36e5ae9#egg=PyGSP",
                     "healpy>=1.15.0",
                     ]

setup(
    author="Esteban Vaca",
    author_email='e.vaca.cerda@fz-juelich.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Spherical Invertible Neural Networks.",
    entry_points={
        'console_scripts': [
            'spherical_inns=spherical_inns.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='spherical_inns',
    name='spherical_inns',
    packages=find_packages(include=['spherical_inns', 'spherical_inns.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/estebvac/spherical_inns',
    version='0.1.0',
    zip_safe=False,
)
