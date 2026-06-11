from setuptools import setup, find_packages

setup(
    name='rtrend_tools',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'colorama',
        'gh',
        'numba',
        'pathos',
        'matplotlib',
        'requests'
        # Add other dependencies if your code requires them
    ],
    author='utkarsh',
    description='Tools for baseline mosquito abundance forecasting and R(t) modeling.',
    include_package_data=True,
)
