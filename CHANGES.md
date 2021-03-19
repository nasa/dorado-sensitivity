# Changelog

## Version 0.3.0 (2021-03-18)

-   The ``get_limmag`` function now accepts ``synphot.SourceSpectrum``
    instance, rather than a spectral model class, for more consistency with the
    other methods and for more flexibility.

## Version 0.2.0 (2021-03-03)

-   Update the instrument characteristics with the values in the Dorado concept
    study report. The effective area and the spot size both increased slightly.

-   Remove the ``bandpass`` argument for convenience, because Dorado has a
    single filter.

## Version 0.1.0 (2021-01-11)

-   First public release.
