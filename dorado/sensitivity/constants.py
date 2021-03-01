#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Scalar constants."""
from astropy import units as u
import numpy as np

AREA = 100.0 * u.cm**2
"""Fiducial collecting area.

Notes
-----
This is a fictitious area that is only used for converting effective area to
synphot dimensionless units. It is only equal to the actual collecting area of
the telescope to within an order of magnitude.
"""

PLATE_SCALE = 25.0 * u.arcsec * u.pix**-1
"""Plate scale.

Notes
-----
Assuing a fixed focal length and 2x2 binning.
"""

NPIX = np.pi * 0.89**2
"""Effective number of pixels in a circular aperture.

Notes
-----
26.8 um = 22.3" = 0.89 pix
"""

APERTURE_CORRECTION = 0.7
"""Aperture correction.

70% of the signal from a point source falls within the above circular aperture.
"""

DARK_NOISE = 4. * 30. * 0.124 * u.hour**-1
"""
Dark noise rate.

Notes
-----
From JPL spreadsheet; 4x for binning; 30x for MDF.
"""

READ_NOISE = 5.0
"""
Read noise.

Notes
-----
From CBE in proposal.
"""
