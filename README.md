# Dorado sensitivity and exposure time calculator

Dorado is a proposed space mission for ultraviolet follow-up of gravitational
wave events. This repository contains a simple sensitivity and exposure time
calculator for Dorado.

This package can estimate the signal to noise, exposure time, or limiting
magnitude for an astronomical source with a given spectrum using the [CCD
signal to noise equation]. It models the following noise contributions:

*   Zodiacal light
*   Airglow (geocoronal emission)
*   Standard CCD noise (shot noise, read noise, dark current)

## Installation

To install with [Pip]:

    $ pip install dorado-sensitivity

## Examples

For examples, see the [Jupyter notebook].

## Dependencies

*   [Astropy]
*   [Synphot] for combining bandpasses and source spectra
*   [PyYAML] for reading [ECSV] data files

[CCD signal to noise equation]: https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-4-computing-exposure-times
[Pip]: https://pip.pypa.io
[Astropy]: https://www.astropy.org
[Synphot]: https://synphot.readthedocs.io/
[PyYAML]: https://pyyaml.org/
[ECSV]: https://github.com/astropy/astropy-APEs/blob/master/APE6.rst
[Jupyter notebook]: https://github.com/nasa/dorado-sensitivity/blob/master/example.ipynb
