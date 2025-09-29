<!-- markdownlint-disable MD041 -->
<!--
A new scriv changelog fragment.

Uncomment the section that is right (remove the HTML comment wrapper).
-->

### Removed

- Removed `DataValues` and `DataValuesCentral` classes. Addition of
  `resample_values` option to `DataCentralMomentsVals`, and addition of
  `Dataset` support from `cmomy` package, made these classes redundant.

### Added

- Added `resample_values` option to `DataCentralMomentsVals` class. This allows
  resampling on `uv` and `xv` instead of resampling during construction of
  `dxduave`. Used in `PerturbModel`, etc.

<!--
### Changed

- A bullet item for the Changed category.

-->
<!--
### Deprecated

- A bullet item for the Deprecated category.

-->
<!--
### Fixed

- A bullet item for the Fixed category.

-->
<!--
### Security

- A bullet item for the Security category.

-->
