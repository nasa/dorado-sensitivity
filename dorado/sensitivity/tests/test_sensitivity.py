#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from hypothesis import given, settings
from hypothesis.strategies import booleans, floats, just
from synphot import ConstFlux1D, SourceSpectrum
from pytest import approx

from .. import get_exptime, get_limmag, get_snr


@given(
    snr=floats(1, 1e9),
    exptime=floats(1, 1e9),
    ra=just(0),
    dec=just(90),
    time=just(Time('2020-01-01')),
    night=booleans())
@settings(deadline=None)
def test_round_trip_snr_limmag(snr, exptime, ra, dec, time, night):
    """Test round trip: get_snr(get_limmag(...))"""
    kwargs = dict(
        exptime=exptime * u.s,
        coord=SkyCoord(ra * u.deg, dec * u.deg),
        time=time, night=night)

    limmag = get_limmag(
        SourceSpectrum(ConstFlux1D, amplitude=0*u.ABmag), snr=snr, **kwargs)
    snr_2 = get_snr(SourceSpectrum(ConstFlux1D, amplitude=limmag), **kwargs)
    assert snr_2 == approx(snr)


@given(
    snr=floats(1, 1e9),
    mag=floats(15, 22),
    ra=just(0),
    dec=just(90),
    time=just(Time('2020-01-01')),
    night=booleans())
@settings(deadline=None)
def test_round_trip_snr_exptime(snr, mag, ra, dec, time, night):
    """Test round trip: get_snr(get_limmag(...))"""
    source = SourceSpectrum(ConstFlux1D, amplitude=mag * u.ABmag)
    kwargs = dict(
        coord=SkyCoord(ra * u.deg, dec * u.deg),
        night=night, time=time)

    exptime = get_exptime(source, snr=snr, **kwargs)
    snr_2 = get_snr(source, exptime=exptime, **kwargs)
    assert snr_2 == approx(snr)
