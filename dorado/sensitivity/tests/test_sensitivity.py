from astropy.coordinates import SkyCoord
from astropy import units as u
from hypothesis import given
from hypothesis.strategies import booleans, floats, just
from synphot import ConstFlux1D, SourceSpectrum
from pytest import approx

from .. import get_limmag, get_snr


@given(
    snr=floats(1, 1e9),
    exptime=floats(1, 1e9),
    ra=just(0),
    dec=just(90),
    night=booleans(),
    bandpass=just('D1'))
def test_round_trip(snr, exptime, ra, dec, night, bandpass):
    """Test round trip: get_snr(get_limmag(...))"""
    kwargs = dict(
        exptime=exptime * u.s,
        coord=SkyCoord(ra * u.deg, dec * u.deg),
        night=night, bandpass=bandpass)

    limmag = get_limmag(ConstFlux1D, snr=snr, **kwargs)
    snr_2 = get_snr(SourceSpectrum(ConstFlux1D, amplitude=limmag), **kwargs)
    assert snr_2 == approx(snr)
