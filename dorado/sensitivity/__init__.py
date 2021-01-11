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
from synphot import Observation, SourceSpectrum

from . import backgrounds
from . import bandpasses
from . import constants

__all__ = ('get_snr', 'get_limmag', 'get_exptime')


def _get_count_rate(source_spectrum, bandpass):
    observation = Observation(source_spectrum, bandpass)
    try:
        return observation.countrate(constants.AREA) / u.ct
    except SynphotError as e:
        if e.args[0] == 'Integrated flux is infinite':
            return np.inf * u.s**-1
        else:
            raise


def get_snr(source_spectrum, *, exptime, coord, time, night, bandpass='D1'):
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
    bandpass : synphot.SpectralElement
        The bandpass (default: D1 filter)

    Returns
    -------
    float
        The signal to noise ratio
    """
    if bandpass == 'D1':
        bandpass = bandpasses.D1

    return signal_to_noise_oir_ccd(
        exptime,
        constants.APERTURE_CORRECTION * _get_count_rate(source_spectrum, bandpass),
        (
            _get_count_rate(backgrounds.get_zodiacal_light(coord, time), bandpass) +
            _get_count_rate(backgrounds.get_airglow(night), bandpass)
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


def get_limmag(model, *, snr, exptime, coord, time, night, bandpass='D1'):
    """Get the limiting magnitude for a given SNR.

    Parameters
    ----------
    source_model : synphot.Model
        The spectral model of the source.
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
    bandpass : synphot.SpectralElement
        The bandpass (default: D1 filter)

    Returns
    -------
    astropy.units.Quantity
        The AB magnitude of the source
    """
    if bandpass == 'D1':
        bandpass = bandpasses.D1

    result = _amp_for_signal_to_noise_oir_ccd(
        snr,
        exptime,
        constants.APERTURE_CORRECTION * _get_count_rate(SourceSpectrum(model, amplitude=0*u.ABmag), bandpass),
        (
            _get_count_rate(backgrounds.get_zodiacal_light(coord, time), bandpass) +
            _get_count_rate(backgrounds.get_airglow(night), bandpass)
        ),
        constants.DARK_NOISE,
        constants.READ_NOISE,
        constants.NPIX
    ).to_value(u.dimensionless_unscaled)

    return -2.5 * np.log10(result) * u.ABmag


def _exptime_for_signal_to_noise_oir_ccd(
        snr, source_eps, sky_eps, dark_eps, rd, npix, gain=1.0):
    """Inverse of astropy.stats.signal_to_noise_oir_ccd."""
    c1 = source_eps * gain
    c2 = npix * (sky_eps * gain + dark_eps)
    c3 = npix * np.square(rd)
    x = 1 + c2 / c1
    snr2 = np.square(snr)
    return 0.5 * snr2 / c1 * (x + np.sqrt(np.square(x) + 4 * c3 / snr2))


def get_exptime(source_spectrum, *, snr, coord, time, night, bandpass='D1'):
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
    bandpass : synphot.SpectralElement
        The bandpass (default: D1 filter)

    Returns
    -------
    astropy.units.Quantity
        The exposure time
    """
    if bandpass == 'D1':
        bandpass = bandpasses.D1

    return _exptime_for_signal_to_noise_oir_ccd(
        snr,
        constants.APERTURE_CORRECTION * _get_count_rate(source_spectrum, bandpass),
        (
            _get_count_rate(backgrounds.get_zodiacal_light(coord, time), bandpass) +
            _get_count_rate(backgrounds.get_airglow(night), bandpass)
        ),
        constants.DARK_NOISE,
        constants.READ_NOISE,
        constants.NPIX
    )
