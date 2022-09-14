import numpy as np


class Parameters:
    # TODO: Include each parameter in docstring
    def __init__(self, step_size=1, distribution=None, months=None,
                 lower_bound=None, lower_threshold=None, upper_bound=None, upper_threshold=None,
                 n_iterations=0, halfwin_ubc=0, trend_preservation='additive', n_quantiles=50,
                 p_value_eps=1.e-10, max_change_factor=100., max_adjustment_factor=9.,
                 if_all_invalid_use=None, adjust_p_values=False, detrend=False, unconditional_ccs_transfer=False,
                 trendless_bound_frequency=False, repeat_warnings=False, invalid_value_warnings=False, adjust_obs=True):
        """
        Initializes the bias adjustment parameters object

        Parameters
        ----------
        trend_preservation: str, optional
            Kind of trend preservation:
            'additive'       # Preserve additive trend.
            'multiplicative' # Preserve multiplicative trend, ensuring
                             # 1/max_change_factor <= change factor
                             #                     <= max_change_factor.
            'mixed'          # Preserve multiplicative or additive trend or mix of
                             # both depending on sign and magnitude of bias. Purely
                             # additive trends are preserved if adjustment factors
                             # of a multiplicative adjustment would be greater than
                             # max_adjustment_factor.
            'bounded'        # Preserve trend of bounded variable. Requires
                             # specification of lower_bound and upper_bound. It is
                             # ensured that the resulting values stay within these
                             # bounds.
        n_quantiles: int, optional
            Number of quantile-quantile pairs used for non-parametric quantile
            mapping.
        max_change_factor: float, optional
            Maximum change factor applied in non-parametric quantile mapping with
            multiplicative or mixed trend preservation.
        max_adjustment_factor: float, optional
            Maximum adjustment factor applied in non-parametric quantile mapping
            with mixed trend preservation.
        adjust_obs: boolean, optional
            If True then transfer simulated climate change signal to x_obs_hist,
            otherwise apply non-parametric quantile mapping to x_sim_fut.
        lower_bound: float, optional
            Lower bound of values in x_obs_hist, x_sim_hist, and x_sim_fut. Used
            for bounded trend preservation.
        upper_bound: float, optional
            Upper bound of values in x_obs_hist, x_sim_hist, and x_sim_fut. Used
            for bounded trend preservation.
        adjust_obs: boolean, optional
            If True then transfer simulated climate change signal to x_obs_hist,
            otherwise apply non-parametric quantile mapping to x_sim_fut.
        """
        # Setting the parameters for the adjustment
        self.step_size = step_size
        self.months = months if not step_size else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.distribution = distribution
        self.trendless_bound_frequency = trendless_bound_frequency
        self.unconditional_ccs_transfer = unconditional_ccs_transfer
        self.detrend = detrend
        self.adjust_p_values = adjust_p_values
        self.if_all_invalid_use = if_all_invalid_use
        self.max_adjustment_factor = max_adjustment_factor
        self.max_change_factor = max_change_factor
        self.p_value_eps = p_value_eps
        self.n_quantiles = n_quantiles
        self.trend_preservation = trend_preservation
        self.halfwin_ubc = halfwin_ubc
        self.n_iterations = n_iterations
        self.upper_threshold = upper_threshold
        self.upper_bound = upper_bound
        self.lower_threshold = lower_threshold
        self.lower_bound = lower_bound
        self.adjust_obs = adjust_obs

        # Quality of life params
        self.invalid_value_warnings = invalid_value_warnings
        self.repeat_warnings = repeat_warnings

        # Assert that parameters make sense
        if step_size:
            self.assert_validity_of_step_size()
        self.assert_validity_of_months()
        self.assert_consistency_of_bounds_and_thresholds()
        self.assert_consistency_of_distribution_and_bounds()

    def assert_validity_of_step_size(self):
        """
        Raises an assertion error if step_size is not an uneven integer between 1
        and 31.
        """
        step_sizes_allowed = np.arange(1, 32, 2)
        msg = 'step_size has to be equal to 0 or an uneven integer between 1 and 31'
        assert self.step_size in step_sizes_allowed, msg

    def assert_validity_of_months(self):
        """
        Raises an assertion error if any of the numbers in months is not in
        {1,...,12}.
        """
        months_allowed = np.arange(1, 13)
        for month in self.months:
            assert month in months_allowed, f'found {month} in months'

    def assert_consistency_of_bounds_and_thresholds(self):
        """
        Raises an assertion error if the pattern of specified and
        unspecified bounds and thresholds is not valid or if
        lower_bound < lower_threshold < upper_threshold < upper_bound
        does not hold.
        """
        lower = self.lower_bound is not None and self.lower_threshold is not None
        upper = self.upper_bound is not None and self.upper_threshold is not None

        if not lower:
            msg = 'lower_bound is not None and lower_threshold is None'
            assert self.lower_bound is None, msg
            msg = 'lower_bound is None and lower_threshold is not None'
            assert self.lower_threshold is None, msg
        if not upper:
            msg = 'upper_bound is not None and upper_threshold is None'
            assert self.upper_bound is None, msg
            msg = 'upper_bound is None and upper_threshold is not None'
            assert self.upper_threshold is None, msg

        if lower:
            assert self.lower_bound < self.lower_threshold, 'lower_bound >= lower_threshold'
        if upper:
            assert self.upper_bound > self.upper_threshold, 'upper_bound <= upper_threshold'
        if lower and upper:
            msg = 'lower_threshold >= upper_threshold'
            assert self.lower_threshold < self.upper_threshold, msg

    def assert_consistency_of_distribution_and_bounds(self):
        """
        Raises an assertion error if the distribution is not consistent with the
        pattern of specified and unspecified bounds and thresholds.
        """
        if self.distribution is not None:
            lower = self.lower_bound is not None and self.lower_threshold is not None
            upper = self.upper_bound is not None and self.upper_threshold is not None

            msg = self.distribution + ' distribution '
            if self.distribution == 'normal':
                assert not lower and not upper, msg + 'can not have bounds'
            elif self.distribution in ['weibull', 'gamma', 'rice']:
                assert lower and not upper, msg + 'must only have lower bound'
            elif self.distribution == 'beta':
                assert lower and upper, msg + 'must have lower and upper bound'
            else:
                raise AssertionError(msg + 'not supported')
