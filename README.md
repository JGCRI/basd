# Version of basd that was used to generate output for use in Xanthos in Jan 2023

# basd

#### `basd` is an open-source Python package for bias adjustment and statistical downscaling of climate model output.

### Purpose
`basd` was created to:

  - Bias adjust climate variables at specific locations or by global grid,

  - Downscale simulated model outcomes to the resolution of observational data,

  - Provide tools for visualization, cross validation, and other tools to better understand methods.

### Getting Started

#### Installation

1. Clone this repo: 

        git clone https://github.com/JGCRI/basd

2. Set up Python virtual environment, for example using conda:

        conda create -n envbasd python=3.9 anaconda
   Though the name of the environment and exact Python version can differ. Then you'd have to activate the
   environment with

        conda activate envbasd
3. Navigate to the cloned repository:

        cd /<path to directory>/basd
   Install the `basd` package using developer mode:

        python setup.py develop

#### Using `basd`
Once you've clone the repo and installed the package, take a look at the **quickstarter.ipynb** in the
**notebooks** directory. This uses a data example included in the repo to show you around the package.

### Documentation
User guide and details on the method can be found on the [Wiki page](https://github.com/JGCRI/basd/wiki). 
Links to further references can be found on those pages.

