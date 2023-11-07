Quickstarter
============

In this example, we'll use the test precipitation data supplied in the ``basd`` repo
to see how to generate bias adjusted and statistical downscaled output.

Loading Packages
----------------

Start by importing ``basd``, ``xarray``, ``dask``, and some other utility packages.

.. code:: ipython3
    
    import os
    import pkg_resources

    import basd
    from dask.distributed import Client, LocalCluster
    import xarray as xr

Reading Data
------------

Then define the paths to our input data. For input into ``basd`` you need three datasets:

1. A reference dataset. This is an observational dataset over a past period (we'll call it the reference period), 
   used to as reference for the dataset we want to adjust/downscale. 
2. A simulated dataset over the reference period. This could be a CMIP6 dataset for example.
3. The same or associated simulated dataset some other period (the application period).

Locate the test precipitation data and define the paths:

.. code:: ipython3

    input_dir = pkg_resources.resource_filename('basd', 'data')
    output_dir = pkg_resources.resource_filename('basd', 'data/output')

and read in the data using ``xarray``:

.. code:: ipython3

    pr_obs_hist = xr.open_mfdataset(os.path.join(input_dir, 'pr_obs-hist_fine_1979-2014.nc'), chunks={'time': 100})
    pr_sim_hist = xr.open_mfdataset(os.path.join(input_dir, 'pr_sim-hist_coarse_1979-2014.nc'), chunks={'time': 100})
    pr_sim_fut = xr.open_mfdataset(os.path.join(input_dir, 'pr_sim-fut_coarse_2065-2100.nc'), chunks={'time': 100})

If you're familiar with ``xarray`` you may notice the ``xr.open_mfdataset`` command to be slightly
different than the default option. What we've done is lazily loaded the data using ``dask``
and ``dask.array`` as the backend data structure in which we've loaded into. This allows
us to load data and perform computations only when we're ready, which we need when working
with large datasets. We've also "chunked" the data along the time dimension at every 100th
data-point. This is breaking up our data into smaller pieces, which again helps to not load
a large amount of data at once, but also to parallelize the processes that we have coming up.

Using Dask Distributed 
----------------------

As mentioned, ``basd`` makes use of the ``dask`` package, including using distributed computing.
Users are free to edit the workflow for what works best for them and their machine. However,
because ``basd`` has some GIL locked dependencies, make sure to use multi-processing.
A basic set-up looks like this:

.. code:: ipython3

    cluster = LocalCluster(processes=True, threads_per_worker=1)
    client = Client(cluster)

which creates a number of processes and workers depending on the core count of your machine
which will be accessible to ``dask`` to distribute tasks to perform. 

We also suggest using the ``with`` utility for better management of closing the cluster at the
end of your task, or if any errors occur. You will see that used below.

Initializing Bias Adjustment 
----------------------------

We initialize our bias adjustment process by feeding ``basd`` our input data and 
other parameters. It then does some pre-processing to make sure the data and parameters
we supplied are valid inputs.

First let's set our parameters by creating a ``basd.Parameters`` object:

.. code:: ipython3

    params = basd.Parameters(
        lower_bound=0, lower_threshold=0.0000011574,
        trend_preservation='mixed',
        distribution='gamma',
        if_all_invalid_use=0, n_iterations=20
        )

The exact settings of your parameters object will be different for different climate variables.
The choices made here will be discussed elsewhere, and we will provide default options for you
for different variables based on literature, though you're welcome to change values as suits your needs.

Then we can pass our parameters and input data to our initialization function:

.. code:: ipython3

    ba = basd.init_bias_adjustment(
        pr_obs_hist, pr_sim_hist, pr_sim_fut,
        'pr', params, 1, 1
    )

Running Bias Adjustment
-----------------------

.. code:: ipython3

    basd.adjust_bias(
        init_output = ba, output_dir = output_dir,
        day_file = ba_day_output_file, month_file = ba_mon_output_file,
        clear_temp = True, encoding={ 'pr': coarse_encoding}
    )
