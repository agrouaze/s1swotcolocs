# s1swotcolocs

[![PyPI version](https://img.shields.io/pypi/v/s1swotcolocs.svg)](https://pypi.python.org/pypi/s1swotcolocs)
[![Build Status](https://img.shields.io/travis/agrouaze/s1swotcolocs.svg)](https://travis-ci.com/agrouaze/s1swotcolocs)
[![Documentation Status](https://readthedocs.org/projects/s1swotcolocs/badge/?version=latest)](https://s1swotcolocs.readthedocs.io/en/latest/?version=latest)
[![Updates](https://pyup.io/repos/github/agrouaze/s1swotcolocs/shield.svg)](https://pyup.io/repos/github/agrouaze/s1swotcolocs/)

Python lib to create co-locations between Sentinel-1 IW or EW images and SWOT KaRin swath.

-   Free software: MIT license
-   Documentation: [https://s1swotcolocs.readthedocs.io](https://s1swotcolocs.readthedocs.io).

# Features

---

-   Find temporal and spatial co-locations between Sentinel-1 (S1) Level-1/Level-2 products and SWOT Level-3 (L3) SSH data.
-   Process S1 IW (Interferometric Wide swath) and EW (Extra Wide swath) modes.
-   Interface with CDSE (Copernicus Data Space Ecosystem) for S1 data discovery.
-   Generate output defining co-located S1 and SWOT data segments.
-   Configurable time delta for co-location criteria.
-   Utility functions for geospatial operations and data handling relevant to S1 and SWOT.

# usage

```python
   import s1swotcolocs
```

# alias

## creating meta-data colocation files

```bash
# to use the lib within a docker image
./coloc_SWOT_L3_with_S1_CDSE_TOPS_sequential_wrapper.py --startdate 20250616 --stopdate 20250616 --confpath src/s1swotcolocs/localconfig.yml

# to use the lib
coloc_SWOT_L3_with_S1_CDSE_TOPS_sequentiel --startmonth 20250616 --stopmonth 20250616 --confpath src/s1swotcolocs/localconfig.yml --outputdir /tmp/
```
