import pandas as pd
import seaborn as sns
import xarray as xr

import basd.ba_params as par


class BaLocOutput:
    def __init__(self, sim_fut_ba: xr.DataArray, sim_fut: xr.DataArray,
                 variable: str, params: par.Parameters):
        # Turn DataArrays into DataFrames
        ba_df = sim_fut_ba.to_dataframe().reset_index()
        df = sim_fut.to_dataframe().reset_index()

        # Add column to signify the source of the time series
        ba_df['Source'] = 'Simulated Future Bias Adjusted'
        df['Source'] = 'Simulated Future'

        # Concatenate into full DataFrame
        self.time_series = pd.concat([ba_df, df], ignore_index=True)

        # Set basic attributes
        self.params = params
        self.variable = variable

    def plot_hist(self, scale=None, bins=15, palette='light:m_r', style='white', xlab=None,
                  title=None):
        sns.set_style(style=style)
        p = sns.histplot(data=self.time_series, x=self.variable, hue='Source', bins=bins,
                         multiple='dodge', palette=palette)
        if scale:
            p.set_yscale(scale)
        if xlab:
            p.set_xlabel(xlab)
        if title:
            p.set_title(title)
