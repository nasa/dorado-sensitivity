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

__all__ = ('NUV_D',)


def _get_bandpass(name):
    with resources.path(data, f'{name}_effective_area.ecsv') as p:
        t = QTable.read(p)
    return SpectralElement(Empirical1D,
                           points=t['wavelength'],
                           lookup_table=t['effective_area'] / constants.AREA)


NUV_D = _get_bandpass('nuv_d')
