# Changelog

## [Unreleased]

### Changed

- **Breaking**: angles are the polar angle and azimuth of the particle velocity direction; `plasma_moments` previously took the latitude (SPEDAS), pitch-angle functions look directions.
- **Breaking**: energy and angle bins are an `EnergyAngleBins`; `sc_pot` and `B` are keywords; `plasma_moments` requires `species` and `units`, and a custom `species` is a PlasmaPopulations.jl `Particle`.
- **Breaking**: a time series is a vector of scans (e.g. `eachslice`) with per-scan inputs as vectors, replacing `tdim`/`edim` and the `t*` functions.
- **Breaking**: `tmoments` → `plasma_moments`; `pitch_angle_distribution`, `tpitch_angle_distribution` → `angle_distribution`; `directional_energy_spectra` → `directional_spectra`; `PAspectra` → `group_mean`.
- **Breaking**: `angle_distribution` `:mean` is solid-angle weighted, empty `:sum` bins are `NaN`, and bins are returned as `edges` and `centers`.
- **Breaking**: `qflux` integrates each bin's solid angle exactly (up to 2% from SPEDAS for MMS FPI ions); species masses are CODATA values; moments are at least `Float64`; `mftens` and `ptens` are 3×3 matrices instead of `(xx, yy, zz, xy, xz, yz)`.

### Removed

- **Breaking**: `sort_flux_by_pitch_angle!`, `tmoments(dists::AbstractVector, ...)`.
- **Breaking**: analytic distributions (`Maxwellian`, `Kappa`, `VelocityDistribution`, ...); use [PlasmaPopulations.jl](https://github.com/JuliaPlasma/PlasmaPopulations.jl).

## [0.2.0] - 2025-12-12

### Changed

- **Breaking**: Drift velocity is no longer stored in the core PDF types (`MaxwellianPDF`, `KappaPDF`, `BiMaxwellianPDF`, `BiKappaPDF`).
  Drift is now handled by the public constructors via keyword arguments `u0` (e.g. `Kappa(...; u0=...)`).

- Distribution constructors support an `n` first argument by returning `VelocityDistribution(n, ...)`.

### Added

- `ShiftedPDF` wrapper for representing drifted distributions without duplicating drift logic across each PDF.

[Unreleased]: https://github.com/JuliaSpacePhysics/VelocityDistributionFunctions.jl/compare/v0.2.0...HEAD
