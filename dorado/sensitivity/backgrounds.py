"""Sky backgrounds."""
from importlib import resources
from warnings import warn

from astropy.coordinates import (GeocentricMeanEcliptic,
                                 HeliocentricMeanEcliptic,
                                 SkyCoord)
from astropy.coordinates.errors import UnitsError
from astropy.table import QTable
from astropy import units as u
import numpy as np
from numpy import inf
from scipy.interpolate import interp2d
from synphot import Empirical1D, GaussianFlux1D, SourceSpectrum

from . import constants
from . import data

__all__ = ('get_zodiacal_light', 'get_airglow')

# Zodiacal light angular dependence from Table 6.2 of
# https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-5-detector-and-sky-backgrounds
lat = np.arange(0, 105, 15)
lon = np.arange(180, -15, -15)
sb = np.asarray([[22.1, 22.4, 22.7, 23.0, 23.2, 23.4, 23.3],
                 [22.3, 22.5, 22.8, 23.0, 23.2, 23.4, 23.3],
                 [22.4, 22.6, 22.9, 23.1, 23.3, 23.4, 23.3],
                 [22.4, 22.6, 22.9, 23.2, 23.3, 23.4, 23.3],
                 [22.4, 22.6, 22.9, 23.2, 23.3, 23.3, 23.3],
                 [22.2, 22.5, 22.9, 23.1, 23.3, 23.3, 23.3],
                 [22.0, 22.3, 22.7, 23.0, 23.2, 23.3, 23.3],
                 [21.7, 22.2, 22.6, 22.9, 23.1, 23.2, 23.3],
                 [21.3, 21.9, 22.4, 22.7, 23.0, 23.2, 23.3],
                 [-inf, -inf, 22.1, 22.5, 22.9, 23.1, 23.3],
                 [-inf, -inf, -inf, 22.3, 22.7, 23.1, 23.3],
                 [-inf, -inf, -inf, -inf, 22.6, 23.0, 23.3],
                 [-inf, -inf, -inf, -inf, 22.6, 23.0, 23.3]])
_stis_zodi_angular_dependence = interp2d(lat, lon, sb, bounds_error=True)
_stis_zodi_high_coord = SkyCoord(180*u.deg, 0*u.deg,
                                 frame=HeliocentricMeanEcliptic)
del lat, lon, sb

# Read zodiacal light spectrum
with resources.path(data, 'stis_zodi_high.ecsv') as p:
    data = QTable.read(p)
_stis_zodi_high = SourceSpectrum(
    Empirical1D,
    points=data['wavelength'],
    lookup_table=data['surface_brightness'] * u.arcsec**2)
del data


def _get_zodi_angular_dependence(coord):
    try:
        coord = SkyCoord(coord).transform_to(HeliocentricMeanEcliptic)
    except UnitsError:
        warn('Supplied coordinates do not have a distance; '
             'asssuming they describe a fixed star for purpose of conversion '
             'to heliocentric coordinates')
        coord = SkyCoord(coord).transform_to(GeocentricMeanEcliptic)

    # Wrap angles and look up in table
    lat = np.abs(coord.lat.deg)
    lon = np.abs(coord.lon.wrap_at(180 * u.deg).deg)
    result = _stis_zodi_angular_dependence(lat, lon)

    # When interp2d encounters infinities, it returns nan. Fix that up here.
    result = np.where(np.isnan(result), -np.inf, result)

    # Fix up shape
    if coord.isscalar:
        result = result.item()

    # Done!
    return result


def get_zodiacal_light(coord):
    """Get the zodiacal light spectrum incident on one pixel.

    Estimate the zodiacal light spectrum based on the angular dependence
    (Table 6.2) and wavelength (Table 6.4) from the STIS Instrument Manual.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        The coordinates of the object under observation. If the coordinates do
        not specify a distance, then the object is assumed to be a fixed star
        at infinite distance for the purpose of calculating its helioecliptic
        position.

    Returns
    -------
    synphot.SourceSpectrum
        The zodiacal light spectrum, normalized to one pixel.

    References
    ----------
    https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-5-detector-and-sky-backgrounds

    """
    zodi_mag = _get_zodi_angular_dependence(coord)
    zodi_mag_high = _get_zodi_angular_dependence(_stis_zodi_high_coord)
    scale = u.mag(1).to_physical(zodi_mag - zodi_mag_high)
    scale *= ((constants.PLATE_SCALE * u.pix)**2).to_value(u.arcsec**2)
    return _stis_zodi_high * scale


def get_airglow(night):
    """Get the airglow spectrum incident on one pixel.

    Estimate the zodiacal light spectrum based on the [O II] geocoronal
    emission line (Table 6.5) in the STIS Instrument Manual.

    Parameters
    ----------
    night : bool
        Use the "low" value if True, or the "high" value if False.

    Returns
    -------
    synphot.SourceSpectrum
        The airglow spectrum, normalized to one pixel.

    References
    ----------
    https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-5-detector-and-sky-backgrounds

    """
    flux = 1.5e-17 if night else 3e-15
    flux *= u.erg * u.s**-1 * u.cm**-2 * u.arcsec**-2
    flux *= (constants.PLATE_SCALE * u.pix)**2
    return SourceSpectrum(GaussianFlux1D,
                          mean=2471 * u.angstrom,
                          fwhm=0.023 * u.angstrom,
                          total_flux=flux)
