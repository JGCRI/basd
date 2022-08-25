import re
from setuptools import setup, find_packages


def readme():
    """Return the contents of the project README file."""

    with open('README.md') as f:
        return f.read()


version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", open('basd/__init__.py').read(), re.M).group(1)

setup(
    name='basd',
    version=version,
    packages=find_packages(),
    url='https://github.com/JGCRI/basd',
    license='BSD-2-Clause',
    author='Noah Prime',
    author_email='noah.prime@pnnl.gov',
    description='',
    long_description=readme(),
    long_description_content_type="text/markdown",
    python_requires='>=3.8.*, <4',
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "numpy>=1.23",
        "PyYAML>=6",
        "netCDF4",
        "pandas",
        "scipy",
        "seaborn",
        "xarray"
    ],
    extras_require={
        'dev': [
        ]
    }
)