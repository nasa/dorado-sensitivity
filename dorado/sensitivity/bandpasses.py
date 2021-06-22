#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Bandpass filter curves."""
from astropy.table import QTable
from synphot import Empirical1D, SpectralElement

from importlib import resources

from . import constants
from . import data

__all__ = ('NUV_D', 'NUV_D_CBE', 'NUV_D_BASELINE', 'NUV_D_THRESHOLD')


def _get_bandpass(name):
    with resources.path(data, f'{name}_effective_area.ecsv') as p:
        t = QTable.read(p)
    return SpectralElement(Empirical1D,
                           points=t['wavelength'],
                           lookup_table=t['effective_area'] / constants.AREA)


NUV_D_CBE = _get_bandpass('nuv_d_cbe')
"""Effective area curve current best estimate."""

NUV_D_BASELINE = _get_bandpass('nuv_d_baseline')
"""Effective area curve for the baseline mission."""

NUV_D_THRESHOLD = _get_bandpass('nuv_d_threshold')
"""Effective area curve for the threshold mission."""

NUV_D = NUV_D_CBE
"""Default effective area curve (same as current best estimate)."""
