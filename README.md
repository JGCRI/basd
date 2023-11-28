[![build](https://github.com/JGCRI/basd/actions/workflows/build.yml/badge.svg)](https://github.com/JGCRI/basd/actions/workflows/build.yml)

# basd

`basd` is an open-source Python package for bias adjustment and statistical downscaling of climate model output.

`basd` was created to:

  - Bias adjust climate variables at specific locations or by global grid,

  - Downscale simulated model outcomes to the resolution of observational data,

  - Provide tools for visualization, cross validation, and other tools to better understand methods.

### Getting Started

#### Installation

`basd` requires the use of `conda` and the `xesmf` package. Start by creating a new virtual environment and activate it:

```
conda create --name basd_env
conda activate basd_env
```

Then install `basd` from GitHub:

```
pip install git+https://github.com/JGCRI/basd.git
```

and the `xesmf` package

```
conda install -c conda-forge xesmf
```

#### Using `basd`
Once you've set up your environment, take a look at the **quickstarter.ipynb** in the **notebooks** directory. This uses a data example included in the repo to show you around the package.

Take a look also at the [website](https://jgcri.github.io/basd/index.html) and [Wiki pages](https://github.com/JGCRI/basd/wiki) for more examples and documentation. Links to further references can be found on those pages.

