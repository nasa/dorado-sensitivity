#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Dorado sensitivity calculator"""

from astropy import units as u
import numpy as np

from .core import Profile
from . import data

__all__ = ('threshold', 'baseline', 'cbe')


threshold = Profile(
    name='Threshold',
    dark_noise=0.02 * u.s**-1,
    read_noise=6.0,
    npix=5.9,
    plate_scale=25 * u.arcsec,
    area=np.pi * (0.5 * 13 * u.cm)**2,
    **data.get_bandpass('nuv_d_threshold'))

baseline = Profile(
    name='Baseline',
    dark_noise=0.02 * u.s**-1,
    read_noise=6.0,
    npix=5.9,
    plate_scale=25 * u.arcsec,
    area=np.pi * (0.5 * 13 * u.cm)**2,
    **data.get_bandpass('nuv_d_baseline'))

cbe = Profile(
    name='CBE',
    dark_noise=0.0041 * u.s**-1,
    read_noise=5.0,
    npix=5.1,
    plate_scale=25 * u.arcsec,
    area=np.pi * (0.5 * 13 * u.cm)**2,
    **data.get_bandpass('nuv_d_cbe'))
