============
s1swotcolocs
============


.. image:: https://img.shields.io/pypi/v/s1swotcolocs.svg
        :target: https://pypi.python.org/pypi/s1swotcolocs

.. image:: https://img.shields.io/travis/agrouaze/s1swotcolocs.svg
        :target: https://travis-ci.com/agrouaze/s1swotcolocs

.. image:: https://readthedocs.org/projects/s1swotcolocs/badge/?version=latest
        :target: https://s1swotcolocs.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/agrouaze/s1swotcolocs/shield.svg
     :target: https://pyup.io/repos/github/agrouaze/s1swotcolocs/
     :alt: Updates



Python lib to create co-locations between Sentinel-1 IW or EW images and SWOT KaRin swath.


* Free software: MIT license
* Documentation: https://s1swotcolocs.readthedocs.io.


Features
--------

* Find temporal and spatial co-locations between Sentinel-1 (S1) Level-1/Level-2 products and SWOT Level-3 (L3) SSH data.
* Process S1 IW (Interferometric Wide swath) and EW (Extra Wide swath) modes.
* Interface with CDSE (Copernicus Data Space Ecosystem) for S1 data discovery.
* Generate output defining co-located S1 and SWOT data segments.
* Configurable time delta for co-location criteria.
* Utility functions for geospatial operations and data handling relevant to S1 and SWOT.


::code-block::python
   import s1swotcolocs
