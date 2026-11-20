# Changelog

## [Unreleased]

### Changed

- **Breaking**: `PAspectra` is renamed `pitch_angle_spectra`.
- **Breaking**: angles follow physics spherical coordinates of the particle velocity direction. `plasma_moments`/`tmoments` previously took the latitude `θ` (SPEDAS), now the polar angle. `pitch_angle_distribution`/`tpitch_angle_distribution` previously took look directions, now velocity directions.
- **Breaking**: `plasma_moments` requires explicit `species` and `units` keywords. A custom `species` is a [PlasmaPopulations.jl](https://github.com/JuliaPlasma/PlasmaPopulations.jl) `Particle` instead of `(; mass, charge)`.
- **Breaking**: the energy-angle grid is an `EnergyAngleGrid`, and the spacecraft potential and magnetic field are keywords.
  - `plasma_moments(dist, sc_pot, magf; kw...)` → `plasma_moments(dist.data, EnergyAngleGrid(dist.energy, dist.phi, dist.theta; layout = (:energy, :phi, :theta)); sc_pot, B = magf, kw...)`
  - `tmoments(data, theta, phi, energy, sc_pot, magf; kw...)` → `plasma_moments(data, grid; sc_pot, B, tdim = ndims(data), kw...)`. Time-varying inputs are vectors with one value per time instead of time-first matrices, e.g. `grid = EnergyAngleGrid.(eachrow(energy), eachrow(phi), Ref(theta); layout)`, `B = eachrow(magf)`.
  - `edim` is replaced by the grid's `layout`, naming the axes of one scan, e.g. `layout = (:phi, :theta, :energy)`.
  - Bin widths of a flattened angle axis must be supplied; the estimate was wrong.
- **Breaking**: species masses are CODATA values instead of SPEDAS's approximations.
- **Breaking**: `pitch_angle_distribution` and `tpitch_angle_distribution` are replaced by `angle_distribution(data, grid, coord; B, ...)`, which takes an `EnergyAngleGrid` like `plasma_moments` and also bins by gyrophase, φ or θ. `:mean` is now solid-angle weighted, `:sum` gives `NaN` for bins without finite cells, and bins are returned as `edges` and `centers` instead of `pitch_angles`.
  - `pitch_angle_distribution(vdf, B, φ, θ; kw...)` → `angle_distribution(vdf, EnergyAngleGrid(energy, φ, θ; layout = (:phi, :theta, :energy)), :pitch; B, kw...)`, or `EnergyAngleGrid(nothing, φ, θ; layout = (:phi, :theta))` for 2-D `vdf`
  - `tpitch_angle_distribution(vdf, B, φ, θ; kw...)` with `B[3, time]` → `angle_distribution(vdf, grid, :pitch; B = eachcol(B), tdim = 4, kw...)`; time-varying angles as a vector of grids.
- **Breaking**: `qflux` integrates each bin's solid angle exactly instead of evaluating at bin centers as SPEDAS does; for MMS FPI ions it differs from SPEDAS by up to 2%. Moments are at least `Float64`.

### Removed

- **Breaking**: `tmoments(dists::AbstractVector, sc_pots[, magfs])`; map `plasma_moments` over scans with the corresponding grid and ancillary keywords.
- **Breaking**: the analytic distribution types (`Maxwellian`, `BiMaxwellian`, `Kappa`, ..., their `*PDF` cores, `ShiftedPDF`, `VelocityDistribution`, `AbstractVelocityDistribution`...) are removed. Use [PlasmaPopulations.jl](https://github.com/JuliaPlasma/PlasmaPopulations.jl), with `vth = √(2T/m)` for every model and SI storage:
  - `Maxwellian(vth)` → `BiMaxwellian(; vth)`; `BiMaxwellian(vth_perp, vth_para)` → `BiMaxwellian(; vth_perp, vth_para)`
  - `Kappa(vth, κ)` → `BiKappa(; vth = vth * s, kappa = κ)` with `s = sqrt(κ / (κ - 3 / 2))`, likewise each speed of `BiKappa(vth_perp, vth_para, κ)`
  - `Maxwellian(T)` (default `mass = me`) → `BiMaxwellian(Electron(); T)`, `T` Unitful or in eV
  - `VelocityDistribution(n, pdf)` → `Population(particle, vdf; n)`

## [0.2.0] - 2025-12-12

### Changed

- **Breaking**: Drift velocity is no longer stored in the core PDF types (`MaxwellianPDF`, `KappaPDF`, `BiMaxwellianPDF`, `BiKappaPDF`).
  Drift is now handled by the public constructors via keyword arguments `u0` (e.g. `Kappa(...; u0=...)`).

- Distribution constructors support an `n` first argument by returning `VelocityDistribution(n, ...)`.

### Added

- `ShiftedPDF` wrapper for representing drifted distributions without duplicating drift logic across each PDF.

[Unreleased]: https://github.com/JuliaSpacePhysics/VelocityDistributionFunctions.jl/compare/v0.2.0...HEAD
