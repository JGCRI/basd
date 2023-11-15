.. basd documentation master file, created by
   sphinx-quickstart on Mon Aug 22 18:36:26 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to basd's documentation!
================================

``basd`` is an open-source Python package for bias adjustment and statistical downscaling of climate model output.

``basd`` was created to:
- Bias adjust climate variables at specific locations or by global grid,
- Downscale simulated model outcomes to the resolution of an observational dataset,
- Provide tools for visualization, cross validation, and other tools to better understand methods.

Basics
------

Bias Adjustment
~~~~~~~~~~~~~~~

The method implemented in ``basd`` works with gridded climate models and requires two main inputs; an observational/reference 
dataset, and a simulated dataset split up into two periods. Those two periods will be one that matches the reference dataset, and
the period for which you're targeting for bias adjustment and downscaling.

The resulting output from ``basd`` is a gridded climate dataset of the input variable with trends preserved from the simulated
climate model input, but with a distribution of values which matches the reference in each grid cell.

.. figure:: images/time_series_cubes.png

While the above image, demonstrating the key objects of interest in ``basd`` reference "future"
and "historic" periods, in theory one could use any time period(s) for which they have data.

The figure below shows the result of ``basd``, with the distribution of precipitation values in the given time series of a grid
cell was adjusted to match the observational dataset.

.. figure:: images/ecdfs.png

Statistical Downscaling
~~~~~~~~~~~~~~~~~~~~~~~

For the downscaling process you just need the output from the bias adjustment process, and the same reference dataset.
The downscaling process is actually another bias adjustment task, only this time we're the matching multivariate
distributions of a cluster of fine grid cells within one of the coarse grids cells of the original data.

.. figure:: images/FineInCoarse.png

The process we implement to map a multivariate distribution between to sets of data is stochastic, and hence an
additional improvement to regular interpolation methods which are deterministic and don't very well represent reality.

Additional Readings
-------------------

To learn more about the bias adjustment and statistical downscaling methods used in this package, outside of
any mention of code, visit the ``basd`` github `Wiki page <https://github.com/JGCRI/basd/wiki>`_, and read the 
`inspiring article <https://doi.org/10.5194/gmd-12-3055-2019>`_ by Stefan Lange.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   getting-started/installation
   getting-started/quickstarter

.. toctree::
   :maxdepth: 1
   :caption: Python API

   reference/api
