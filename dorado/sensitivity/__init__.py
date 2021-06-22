#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Dorado sensitivity calculator"""
try:
    from functools import cache
except ImportError:
    # FIXME: remove once we require Python 3.9 and higher.
    # functools.cache was added in Python 3.9, but is implemented using
    # lru_cache with a specific configuration. For our purposes of caching a
    # function that takes no arguments, functools.lru_cache can be used
    # interchangeably.
    from functools import lru_cache as cache

from astropy.stats import signal_to_noise_oir_ccd
from astropy import units as u
import numpy as np
from synphot import Observation
from synphot.units import PHOTLAM

from . import backgrounds
from . import bandpasses
from . import constants

__all__ = ('get_snr', 'get_limmag', 'get_exptime')


def _get_count_rate(source_spectrum, bandpass):
    flux = (source_spectrum * bandpass).integrate(flux_unit=PHOTLAM)
    return flux * (constants.AREA / u.ph)


def _get_background_count_rate(bandpass, coord, time, night):
    airglow_scale = backgrounds.get_airglow_scale(night)
    zodi_scale = backgrounds.get_zodiacal_light_scale(coord, time)
    galactic_scale1, galactic_scale2 = backgrounds.get_galactic_scales(coord)
    night = np.asarray(night)
    return np.square(constants.PLATE_SCALE.to_value(u.arcsec / u.pix)) * (
        _get_count_rate(backgrounds.high_zodiacal_light,
                        bandpass) * zodi_scale +
        _get_count_rate(backgrounds.day_airglow,
                        bandpass) * airglow_scale +
        _get_count_rate(backgrounds.galactic1,
                        bandpass) * galactic_scale1 +
        _get_count_rate(backgrounds.galactic2,
                        bandpass) * galactic_scale2
    )


@cache
def _get_dust_query():
    from dustmaps import planck
    planck.fetch(which='GNILC')
    return planck.PlanckGNILCQuery()


@cache
def _get_reddening_law():
    from dust_extinction.parameter_averages import F19
    from synphot import ReddeningLaw
    return ReddeningLaw(F19())


def _get_reddened_count_rate_scalar(source_spectrum, bandpass, ebv):
    reddening_law = _get_reddening_law()
    extinction_curve = reddening_law.extinction_curve(ebv, bandpass.waveset)
    return _get_count_rate(source_spectrum * extinction_curve, bandpass)


def _get_reddened_count_rate_vector(source_spectrum, bandpass, ebv):
    return u.Quantity([
        _get_reddened_count_rate_scalar(source_spectrum, bandpass, _)
        for _ in np.ravel(ebv)]).reshape(np.shape(ebv))


def _get_reddened_count_rate_slow(source_spectrum, bandpass, ebv):
    if np.isscalar(ebv):
        return _get_reddened_count_rate_scalar(source_spectrum, bandpass, ebv)
    else:
        return _get_reddened_count_rate_vector(source_spectrum, bandpass, ebv)


def _get_reddened_count_rate(source_spectrum, bandpass, ebv):
    steps = 100
    ebv_min = 0.0
    ebv_max = 50.0
    if np.size(ebv) < steps:
        return _get_reddened_count_rate_slow(source_spectrum, bandpass, ebv)
    else:
        from scipy.interpolate import interp1d
        x = np.linspace(ebv_min, ebv_max, steps)
        y = _get_reddened_count_rate_slow(source_spectrum, bandpass, x)
        unit = y.unit
        y = np.log(y.value)
        interp = interp1d(x, y, kind='cubic', assume_sorted=True,
                          bounds_error=False, fill_value=np.inf)
        return np.exp(interp(ebv)) * unit


def _get_source_count_rate(source_spectrum, bandpass, coord, redden):
    if redden:
        dust_query = _get_dust_query()
        ebv = dust_query(coord)
        return _get_reddened_count_rate(source_spectrum, bandpass, ebv)
    else:
        return _get_count_rate(source_spectrum, bandpass)


def get_snr(source_spectrum, *, exptime, coord, time, night, redden=False,
            bandpass=None):
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
    redden : bool
        Whether to apply Milky Way extinction to the source spectrum
    bandpass : synphot.SpecralElement
        Bandpass. Default: Dorado current baseline estimate.

    Returns
    -------
    float
        The signal to noise ratio
    """
    if bandpass is None:
        bandpass = bandpasses.NUV_D
    return signal_to_noise_oir_ccd(
        exptime,
        constants.APERTURE_CORRECTION * _get_source_count_rate(
            source_spectrum, bandpass, coord, redden),
        _get_background_count_rate(bandpass, coord, time, night),
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


def get_limmag(source_spectrum, *, snr, exptime, coord, time, night,
               bandpass=None):
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
    bandpass : synphot.SpecralElement
        Bandpass. Default: Dorado current baseline estimate.

    Returns
    -------
    astropy.units.Quantity
        The AB magnitude of the source
    """
    if bandpass is None:
        bandpass = bandpasses.NUV_D

    mag0 = Observation(source_spectrum, bandpass).effstim(
        u.ABmag, area=constants.AREA)

    result = _amp_for_signal_to_noise_oir_ccd(
        snr,
        exptime,
        constants.APERTURE_CORRECTION * _get_count_rate(
            source_spectrum, bandpass),
        _get_background_count_rate(bandpass, coord, time, night),
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


def get_exptime(source_spectrum, *, snr, coord, time, night, redden=False,
                bandpass=None):
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
    redden : bool
        Whether to apply Milky Way extinction to the source spectrum
    bandpass : synphot.SpecralElement
        Bandpass. Default: Dorado current baseline estimate.

    Returns
    -------
    astropy.units.Quantity
        The exposure time
    """
    if bandpass is None:
        bandpass = bandpasses.NUV_D
    return _exptime_for_signal_to_noise_oir_ccd(
        snr,
        constants.APERTURE_CORRECTION * _get_source_count_rate(
            source_spectrum, bandpass, coord, redden),
        _get_background_count_rate(bandpass, coord, time, night),
        constants.DARK_NOISE,
        constants.READ_NOISE,
        constants.NPIX
    )
