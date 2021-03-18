#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Dorado sensitivity calculator"""
from astropy.stats import signal_to_noise_oir_ccd
from astropy import units as u
import numpy as np
from synphot.exceptions import SynphotError
from synphot import Observation

from . import backgrounds
from . import bandpasses
from . import constants

__all__ = ('get_snr', 'get_limmag', 'get_exptime')


def _get_count_rate(source_spectrum):
    observation = Observation(source_spectrum, bandpasses.NUV_D)
    try:
        return observation.countrate(constants.AREA) / u.ct
    except SynphotError as e:
        if e.args[0] == 'Integrated flux is infinite':
            return np.inf * u.s**-1
        else:
            raise


def get_snr(source_spectrum, *, exptime, coord, time, night):
    """Calculate the SNR of an observation of a point source with Dorado.

    Parameters
    ----------
    source_spectrum : synphot.SourceSpectrum
        The spectrum of the source.
    exptime : astropy.units.Quantity
        The exposure time
    coord : astropy.coordinates.SkyCoord
        The coordinates of the source, for calculating zodiacal light
    time : astropy.time.Time
        The time of the observation, for calculating zodiacal light
    night : bool
        Whether the observation occurs on the day or night side of the Earth,
        for estimating airglow

    Returns
    -------
    float
        The signal to noise ratio
    """
    return signal_to_noise_oir_ccd(
        exptime,
        constants.APERTURE_CORRECTION * _get_count_rate(source_spectrum),
        (
            _get_count_rate(backgrounds.get_zodiacal_light(coord, time)) +
            _get_count_rate(backgrounds.get_airglow(night))
        ),
        constants.DARK_NOISE,
        constants.READ_NOISE,
        constants.NPIX
    ).to_value(u.dimensionless_unscaled)


def _amp_for_signal_to_noise_oir_ccd(
        snr, t, source_eps, sky_eps, dark_eps, rd, npix, gain=1.0):
    """Inverse of astropy.stats.signal_to_noise_oir_ccd."""
    signal = t * source_eps * gain
    # noise squared without signal shot noise term
    snr2 = np.square(snr)
    noise2 = t * (npix * (sky_eps * gain + dark_eps)) + npix * np.square(rd)
    return 0.5 * snr2 / signal * (1 + np.sqrt(1 + 4 * noise2 / snr2))


def get_limmag(source_spectrum, *, snr, exptime, coord, time, night):
    """Get the limiting magnitude for a given SNR.

    Parameters
    ----------
    source_spectrum : synphot.SourceSpectrum
        The spectrum of the source.
    snr : float
        The desired SNR.
    exptime : astropy.units.Quantity
        The exposure time
    coord : astropy.coordinates.SkyCoord
        The coordinates of the source, for calculating zodiacal light
    time : astropy.time.Time
        The time of the observation, for calculating zodiacal light
    night : bool
        Whether the observation occurs on the day or night side of the Earth,
        for estimating airglow

    Returns
    -------
    astropy.units.Quantity
        The AB magnitude of the source
    """
    mag0 = Observation(source_spectrum, bandpasses.NUV_D).effstim(
        u.ABmag, area=constants.AREA)

    result = _amp_for_signal_to_noise_oir_ccd(
        snr,
        exptime,
        constants.APERTURE_CORRECTION * _get_count_rate(source_spectrum),
        (
            _get_count_rate(backgrounds.get_zodiacal_light(coord, time)) +
            _get_count_rate(backgrounds.get_airglow(night))
        ),
        constants.DARK_NOISE,
        constants.READ_NOISE,
        constants.NPIX
    ).to_value(u.dimensionless_unscaled)

    return -2.5 * np.log10(result) * u.mag + mag0


def _exptime_for_signal_to_noise_oir_ccd(
        snr, source_eps, sky_eps, dark_eps, rd, npix, gain=1.0):
    """Inverse of astropy.stats.signal_to_noise_oir_ccd."""
    c1 = source_eps * gain
    c2 = npix * (sky_eps * gain + dark_eps)
    c3 = npix * np.square(rd)
    x = 1 + c2 / c1
    snr2 = np.square(snr)
    return 0.5 * snr2 / c1 * (x + np.sqrt(np.square(x) + 4 * c3 / snr2))


def get_exptime(source_spectrum, *, snr, coord, time, night):
    """Calculate the SNR of an observation of a point source with Dorado.

    Parameters
    ----------
    source_spectrum : synphot.SourceSpectrum
        The spectrum of the source.
    snr : float
        The signal to noise ratio
    coord : astropy.coordinates.SkyCoord
        The coordinates of the source, for calculating zodiacal light
    time : astropy.time.Time
        The time of the observation, for calculating zodiacal light
    night : bool
        Whether the observation occurs on the day or night side of the Earth,
        for estimating airglow

    Returns
    -------
    astropy.units.Quantity
        The exposure time
    """
    return _exptime_for_signal_to_noise_oir_ccd(
        snr,
        constants.APERTURE_CORRECTION * _get_count_rate(source_spectrum),
        (
            _get_count_rate(backgrounds.get_zodiacal_light(coord, time)) +
            _get_count_rate(backgrounds.get_airglow(night))
        ),
        constants.DARK_NOISE,
        constants.READ_NOISE,
        constants.NPIX
    )
