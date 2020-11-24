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

NPIX = np.pi * 0.80**2 * 0.70
"""Effective number of pixels in a circular aperture.

Notes
-----
For simplicity, use the current CBE spot radius = 24.1um
and 70% of energy within this radius.

From DesignComparisons-20201008.sbc.xls,
24.1 um = 20.1" = 0.80 pix
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
