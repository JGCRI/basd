import numpy as np


def assert_validity_of_months(months):
    """
    Raises an assertion error if any of the numbers in months is not in
    {1,...,12}.

    Parameters
    ----------
    months: np.Array
        Sequence of ints representing calendar months.
    """
    months_allowed = np.arange(1, 13)
    for month in months:
        assert month in months_allowed, f'found {month} in months'
