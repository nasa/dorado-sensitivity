"""Dorado sensitivity calculator"""
from astropy.stats import signal_to_noise_oir_ccd
from astropy import units as u
import numpy as np
from synphot.exceptions import SynphotError
from synphot import Observation, SourceSpectrum

from . import constants
from . import bandpasses
from . import backgrounds

__all__ = ('get_snr',)


def _get_count_rate(source_spectrum, bandpass):
    observation = Observation(source_spectrum, bandpass)
    try:
        return observation.countrate(constants.AREA) / u.ct
    except SynphotError as e:
        if e.args[0] == 'Integrated flux is infinite':
            return np.inf * u.s**-1
        else:
            raise


def get_snr(source_spectrum, *, exptime, coord, night, bandpass='D1'):
    """Calculate the SNR of an observation of a point source with Dorado.

    Parameters
    ----------
    source_spectrum : synphot.SourceSpectrum
        The spectrum of the source.
    exptime : astropy.units.Quantity
        The exposure time
    coord : astropy.coordinates.SkyCoord
        The coordinates of the source, for calculating zodiacal light
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
        _get_count_rate(source_spectrum, bandpass),
        (
            _get_count_rate(backgrounds.get_zodiacal_light(coord), bandpass) +
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
    noise2 = t * (npix * (sky_eps * gain + dark_eps)) + npix * rd ** 2
    return 0.5 * snr**2 / signal * (1 + np.sqrt(1 + 4 * noise2 / snr**2))


def get_limmag(model, *, snr, exptime, coord, night, bandpass='D1'):
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
        _get_count_rate(SourceSpectrum(model, amplitude=0*u.ABmag), bandpass),
        (
            _get_count_rate(backgrounds.get_zodiacal_light(coord), bandpass) +
            _get_count_rate(backgrounds.get_airglow(night), bandpass)
        ),
        constants.DARK_NOISE,
        constants.READ_NOISE,
        constants.NPIX
    ).to_value(u.dimensionless_unscaled)

    return -2.5 * np.log10(result) * u.ABmag
