#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""The CCD SNR equation and inverses of it in terms of various variables."""

from astropy.stats import signal_to_noise_oir_ccd
import numpy as np

__all__ = ('exptime_oir_ccd', 'signal_to_noise_oir_ccd')


def exptime_oir_ccd(signal_to_noise, source_eps, sky_eps, dark_eps, rd, npix,
                    gain=1.0):
    """Computes the exposure time for a given signal to noise ratio for source
    being observed in the optical/IR using a CCD.

    Parameters
    ----------
    signal_to_noise : float or numpy.ndarray
        Desired signal to noise ratio
    source_eps : float
        Number of electrons (photons) or DN per second in the aperture from the
        source. Note that this should already have been scaled by the filter
        transmission and the quantum efficiency of the CCD. If the input is in
        DN, then be sure to set the gain to the proper value for the CCD.
        If the input is in electrons per second, then keep the gain as its
        default of 1.0.
    sky_eps : float
        Number of electrons (photons) or DN per second per pixel from the sky
        background. Should already be scaled by filter transmission and QE.
        This must be in the same units as source_eps for the calculation to
        make sense.
    dark_eps : float
        Number of thermal electrons per second per pixel. If this is given in
        DN or ADU, then multiply by the gain to get the value in electrons.
    rd : float
        Read noise of the CCD in electrons. If this is given in
        DN or ADU, then multiply by the gain to get the value in electrons.
    npix : float
        Size of the aperture in pixels
    gain : float, optional
        Gain of the CCD. In units of electrons per DN.

    Returns
    -------
    t : float or numpy.ndarray
        CCD integration time in seconds

    See also
    --------
    astropy.stats.signal_to_noise_oir_ccd
    """
    snr2 = np.square(signal_to_noise)
    signal_rate = source_eps * gain
    noise_rate = signal_rate + npix * (sky_eps * gain + dark_eps)
    noise_const = npix * rd ** 2
    a = np.square(signal_rate)
    b = -snr2 * noise_rate
    c = -snr2 * noise_const
    return (-b + np.sqrt(np.square(b) - 4 * a * c)) / (2 * a)
