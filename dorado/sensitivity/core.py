#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#

try:
    from functools import cache
except ImportError:
    # FIXME: remove once we require Python 3.9 and higher.
    # functools.cache was added in Python 3.9, but is implemented using
    # lru_cache with a specific configuration. For our purposes of caching a
    # function that takes no arguments, functools.lru_cache can be used
    # interchangeably.
    from functools import lru_cache as cache
from dataclasses import dataclass

from astropy import units as u
import numpy as np
from synphot import Empirical1D, Observation, SpectralElement

from . import backgrounds as bg
from . import math

__all__ = ('Profile',)


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


@dataclass
class BaseProfile:

    name: str
    """A descriptive name for this mission profile."""

    dark_noise: u.s**-1
    """Dark noise per pixel per unit time."""

    read_noise: float
    """Read noise per pixel."""

    npix: float
    """Sharpness of the point spread function in pixels."""

    plate_scale: u.arcsec
    """Plate scale in arcseconds per pixel."""

    area: u.cm**2
    """Fiducial collecting area (pupil area)."""

    band: SpectralElement
    """System throughput curve.

    This is defined as the effective area curve divided by the fiducial
    collecting area.
    """

    def _get_count_rate(self, source_spectrum):
        obs = Observation(source_spectrum, self.band)
        return obs.countrate(self.area) / u.ct

    def _get_background_count_rate(self, coord, time, night):
        airglow_scale = bg.get_airglow_scale(night)
        zodi_scale = bg.get_zodiacal_light_scale(coord, time)
        gal_scale1, gal_scale2 = bg.get_galactic_scales(coord)
        night = np.asarray(night)
        return np.square(self.plate_scale.to_value(u.arcsec)) * (
            self._get_count_rate(bg.high_zodiacal_light) * zodi_scale +
            self._get_count_rate(bg.day_airglow) * airglow_scale +
            self._get_count_rate(bg.galactic1) * gal_scale1 +
            self._get_count_rate(bg.galactic2) * gal_scale2
        )

    def _get_reddened_count_rate_scalar(self, source_spectrum, ebv):
        red_law = _get_reddening_law()
        ext_curve = red_law.extinction_curve(ebv, self.band.waveset)
        return self._get_count_rate(source_spectrum * ext_curve)

    def _get_reddened_count_rate_vector(self, source_spectrum, ebv):
        return u.Quantity([
            self._get_reddened_count_rate_scalar(source_spectrum, _)
            for _ in np.ravel(ebv)]).reshape(np.shape(ebv))

    def _get_reddened_count_rate_slow(self, source_spectrum, ebv):
        if np.isscalar(ebv):
            return self._get_reddened_count_rate_scalar(source_spectrum, ebv)
        else:
            return self._get_reddened_count_rate_vector(source_spectrum, ebv)

    def _get_reddened_count_rate(self, source_spectrum, ebv):
        steps = 100
        ebv_min = 0.0
        ebv_max = 50.0
        if np.size(ebv) < steps:
            return self._get_reddened_count_rate_slow(source_spectrum, ebv)
        else:
            from scipy.interpolate import interp1d
            x = np.linspace(ebv_min, ebv_max, steps)
            y = self._get_reddened_count_rate_slow(source_spectrum, x)
            unit = y.unit
            y = np.log(y.value)
            interp = interp1d(x, y, kind='cubic', assume_sorted=True,
                              bounds_error=False, fill_value=np.inf)
            return np.exp(interp(ebv)) * unit

    def _get_source_count_rate(self, source_spectrum, coord, redden):
        if redden:
            dust_query = _get_dust_query()
            ebv = dust_query(coord)
            return self._get_reddened_count_rate(source_spectrum, ebv)
        else:
            return self._get_count_rate(source_spectrum)

    def get_snr(self, source_spectrum, *, exptime, coord, time, night,
                redden=False):
        """Calculate the SNR of an observation of a point source.

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
            Whether the observation occurs on the day or night side of the
            Earth, for estimating airglow
        redden : bool
            Whether to apply Milky Way extinction to the source spectrum

        Returns
        -------
        float
            The signal to noise ratio
        """
        return math.signal_to_noise_oir_ccd(
            exptime,
            self._get_source_count_rate(source_spectrum, coord, redden),
            self._get_background_count_rate(coord, time, night),
            self.dark_noise,
            self.read_noise,
            self.npix
        ).to_value(u.dimensionless_unscaled)

    def get_exptime(self, source_spectrum, *, snr, coord, time, night,
                    redden=False):
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
            Whether the observation occurs on the day or night side of the
            Earth, for estimating airglow
        redden : bool
            Whether to apply Milky Way extinction to the source spectrum

        Returns
        -------
        astropy.units.Quantity
            The exposure time
        """
        return math.exptime_oir_ccd(
            snr,
            self._get_source_count_rate(source_spectrum, coord, redden),
            self._get_background_count_rate(coord, time, night),
            self.dark_noise,
            self.read_noise,
            self.npix
        )


class Profile(BaseProfile):

    def __init__(self, *args,
                 area, band=None, wave=None, eff_area=None,
                 **kwargs):
        if wave is not None and eff_area is not None:
            band = SpectralElement(Empirical1D,
                                   points=wave,
                                   lookup_table=eff_area/area)
        super().__init__(*args, area=area, band=band, **kwargs)
