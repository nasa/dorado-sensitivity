#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Sky backgrounds."""
from importlib import resources

from astropy.coordinates import GeocentricTrueEcliptic, get_sun, SkyCoord
from astropy.table import QTable
from astropy import units as u
import numpy as np
from scipy.interpolate import interp2d
from synphot import Empirical1D, GaussianFlux1D, PowerLawFlux1D, SourceSpectrum
from synphot.units import PHOTLAM

from . import data

__all__ = ('get_zodiacal_light', 'get_airglow', 'get_galactic')


def _get_zodi_angular_interp():
    # Zodiacal light angular dependence from Table 16 of Leinert et al. (2017)
    # https://doi.org/10.1051/aas:1998105.
    with resources.path(data, 'leinert_zodi.txt') as p:
        table = np.loadtxt(p)
    lat = table[0, 1:]
    lon = table[1:, 0]
    s10 = table[1:, 1:]

    # The table only extends up to a latitude of 75°. According to the paper,
    # "Towards the ecliptic pole, the brightness as given above is 60 ± 3 S10."
    lat = np.append(lat, 90)
    s10 = np.append(s10, np.tile(60.0, (len(lon), 1)), axis=1)

    # The table is in units of S10: the number of 10th magnitude stars per
    # square degree. Convert to magnitude per square arcsecond.
    sb = 10 - 2.5 * np.log10(s10 / 60**4)

    return interp2d(lat, lon, sb, bounds_error=True)


_zodi_angular_dependence = _get_zodi_angular_interp()


# Read zodiacal light spectrum
with resources.path(data, 'stis_zodi_high.ecsv') as p:
    table = QTable.read(p)
_stis_zodi_high = SourceSpectrum(
    Empirical1D,
    points=table['wavelength'],
    lookup_table=table['surface_brightness'] * u.arcsec**2)
del table


def _get_zodi_angular_dependence(coord, time):
    obj = SkyCoord(coord).transform_to(GeocentricTrueEcliptic(equinox=time))
    sun = get_sun(time).transform_to(GeocentricTrueEcliptic(equinox=time))

    # Wrap angles and look up in table
    lat = np.abs(obj.lat.deg)
    lon = np.abs((obj.lon - sun.lon).wrap_at(180 * u.deg).deg)
    result = _zodi_angular_dependence(lat, lon)

    # When interp2d encounters infinities, it returns nan. Fix that up here.
    result = np.where(np.isnan(result), -np.inf, result)

    # Fix up shape
    if obj.isscalar:
        result = result.item()

    return result - _zodi_angular_dependence(0, 180).item()


def get_zodiacal_light(coord, time):
    """Get the zodiacal light spectrum normalized to 1 square arcsecond.

    Estimate the zodiacal light spectrum based on the angular dependence
    (Table 6.2) and wavelength (Table 6.4) from the STIS Instrument Manual.

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        The coordinates of the object under observation. If the coordinates do
        not specify a distance, then the object is assumed to be a fixed star
        at infinite distance for the purpose of calculating its helioecliptic
        position.
    time : astropy.time.Time
        The time of the observation.

    Returns
    -------
    synphot.SourceSpectrum
        The zodiacal light spectrum, normalized to 1 square arcsecond.

    References
    ----------
    https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-5-detector-and-sky-backgrounds

    """
    scale = u.mag(1).to_physical(_get_zodi_angular_dependence(coord, time))
    return _stis_zodi_high * scale


def get_airglow(night):
    """Get the airglow spectrum, normalized to 1 square arcsecond.

    Estimate the zodiacal light spectrum based on the [O II] geocoronal
    emission line (Table 6.5) in the STIS Instrument Manual.

    Parameters
    ----------
    night : bool
        Use the "low" value if True, or the "average" value if False.

    Returns
    -------
    synphot.SourceSpectrum
        The airglow spectrum, normalized to 1 square arcsecond.

    References
    ----------
    https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-5-detector-and-sky-backgrounds

    """
    flux = np.where(night, 1.5e-17, 1.5e-15) * u.erg * u.s**-1 * u.cm**-2
    return SourceSpectrum(GaussianFlux1D,
                          mean=2471 * u.angstrom,
                          fwhm=0.023 * u.angstrom,
                          total_flux=flux)


def get_galactic(coord):
    """Get the Galactic diffuse emission, normalized to 1 square arcsecond.

    Estimate the Galactic diffuse emission based on the cosecant fits from
    Murthy (2014).

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        The coordinates of the object under observation.

    Returns
    -------
    synphot.SourceSpectrum
        The Galactic diffuse emission spectrum, normalized to 1 square
        arcsecond.

    References
    ----------
    https://doi.org/10.3847/1538-4357/aabcb9

    """
    b = SkyCoord(coord).galactic.b
    csc = 1 / np.sin(b)
    pos = (csc > 0)

    # Constants from Murthy (2014) Table 4.
    # Note that slopes for the Southern hemisphere have been negated to cancel
    # the minus sign in the Galactic latitude.
    fuv_a = np.where(pos, 93.4, -205.5)
    nuv_a = np.where(pos, 257.5, 66.7)
    fuv_b = np.where(pos, 133.2, -401.8)
    nuv_b = np.where(pos, 185.1, -356.3)

    fuv = fuv_a + fuv_b * csc
    nuv = nuv_a + nuv_b * csc
    surf_bright_unit = PHOTLAM * u.steradian**-1 * u.arcsec**2

    # GALEX filter effective wavelengths in angstroms from
    # http://www.galex.caltech.edu/researcher/techdoc-ch1.html#3
    fuv_wave = 1528
    nuv_wave = 2271

    return SourceSpectrum(
        PowerLawFlux1D,
        amplitude=fuv * surf_bright_unit,
        x_0=fuv_wave * u.angstrom,
        alpha=-np.log(nuv / fuv) / np.log(nuv_wave / fuv_wave))
