"""Bandpass filter curves."""
from astropy.table import QTable
from synphot import Empirical1D, SpectralElement

from importlib import resources

from . import constants
from . import data

__all__ = ('D1',)


def _get_bandpass(name):
    with resources.path(data, f'{name}_effective_area.ecsv') as p:
        t = QTable.read(p)
    return SpectralElement(Empirical1D,
                           points=t['wavelength'],
                           lookup_table=t['effective_area'] / constants.AREA)


D1 = _get_bandpass('d1')
