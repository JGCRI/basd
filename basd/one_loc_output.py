import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import xarray as xr

import basd.ba_params as par


class BaLocOutput:
    def __init__(self,
                 sim_fut_ba: xr.DataArray, sim_fut: xr.DataArray,
                 obs_hist: xr.DataArray, sim_hist: xr.DataArray,
                 variable: str, params: par.Parameters):

        # Turn DataArrays into DataFrames
        oh_df = obs_hist.to_dataframe().reset_index()
        sh_df = sim_hist.to_dataframe().reset_index()
        ba_df = sim_fut_ba.to_dataframe().reset_index()
        sf_df = sim_fut.to_dataframe().reset_index()

        # Add column to signify the source of the time series
        oh_df['Source'] = 'Observed Historical'
        sh_df['Source'] = 'Simulated Historical'
        ba_df['Source'] = 'Simulated Future Bias Adjusted'
        sf_df['Source'] = 'Simulated Future'

        # Concatenate into full DataFrame
        self.time_series = pd.concat([ba_df, sf_df, oh_df, sh_df], ignore_index=True)

        # Set basic attributes
        self.params = params
        self.variable = variable

    def plot_hist(self, scale=None, bins=15, palette='light:m_r', style='white', xlab=None,
                  title=None):
        indexes = (self.time_series['Source'] == 'Simulated Future') | (
                    self.time_series['Source'] == 'Simulated Future Bias Adjusted')
        sns.set_style(style=style)
        p = sns.histplot(data=self.time_series.loc[indexes], x=self.variable, hue='Source', bins=bins,
                         multiple='dodge', palette=palette)
        if scale:
            p.set_yscale(scale)
        if xlab:
            p.set_xlabel(xlab)
        if title:
            p.set_title(title)

    def plot_ecdf(self, **kwargs):

        # Getting data within thresholds
        within_thresh = np.ones(len(self.time_series.index))
        if self.params.lower_threshold:
            above = self.params.lower_threshold < self.time_series[self.variable]
            within_thresh = np.logical_and(above, within_thresh)
        if self.params.upper_threshold:
            below = self.params.upper_threshold > self.time_series[self.variable]
            within_thresh = np.logical_and(below, within_thresh)

        # Plot Empirical CDF with plotly
        p = px.ecdf(self.time_series.loc[within_thresh],
                    x=self.variable, color='Source', **kwargs)

        return p
