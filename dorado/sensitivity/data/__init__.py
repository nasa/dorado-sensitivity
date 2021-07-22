#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#

from importlib import resources

from astropy.table import QTable


def get_bandpass(name):
    with resources.path(__package__, f'{name}_effective_area.ecsv') as p:
        t = QTable.read(p)
    return dict(wave=t['wavelength'], eff_area=t['effective_area'])
