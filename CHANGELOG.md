# Changelog

## [Unreleased]

### Removed

- **Breaking**: `tmoments(dists::AbstractVector, sc_pots[, magfs])`; use `StructArray(plasma_moments.(dists, sc_pots, magfs; kw...))`.
- **Breaking**: the analytic distribution types (`Maxwellian`, `BiMaxwellian`, `Kappa`, `BiKappa`, their `*PDF` cores, `ShiftedPDF`, `VelocityDistribution`, `AbstractVelocityPDF`, `AbstractVelocityDistribution`, `KappaDistribution`, `pdf`, `kappa_thermal_speed`, `VPar`, `VPerp`) and the Unitful and Distributions extensions are removed. Use [PlasmaPopulations.jl](https://github.com/JuliaSpacePhysics/PlasmaPopulations.jl), with `vth = √(2T/m)` for every model and SI storage:
  - `Maxwellian(vth)` → `BiMaxwellian(; vth)`; `BiMaxwellian(vth_perp, vth_para)` → `BiMaxwellian(; vth_perp, vth_para)`
  - `Kappa(vth, κ)` → `BiKappa(; vth = vth * s, kappa = κ)` with `s = sqrt(κ / (κ - 3 / 2))`, likewise each speed of `BiKappa(vth_perp, vth_para, κ)`, since the old `vth` was not temperature-preserving
  - `kappa_thermal_speed(vth, κ)` → `PlasmaPopulations.kappa_widths(d)`
  - `Maxwellian(T)` (default `mass = me`) → `BiMaxwellian(Electron(); T)`, `T` Unitful or in eV
  - `VelocityDistribution(n, pdf)` → `Population(particle, vdf; n)`
  - `b0` argument or `u0` keyword → `Oriented(d, b0; u = u0)`
  - `d(𝐯)` → `pdf(d, 𝐯)`
  - `VPar`, `VPerp` marginals have no replacement yet

## [0.2.0] - 2025-12-12

### Changed

- **Breaking**: Drift velocity is no longer stored in the core PDF types (`MaxwellianPDF`, `KappaPDF`, `BiMaxwellianPDF`, `BiKappaPDF`).
  Drift is now handled by the public constructors via keyword arguments `u0` (e.g. `Kappa(...; u0=...)`).

- Distribution constructors support an `n` first argument by returning `VelocityDistribution(n, ...)`.

### Added

- `ShiftedPDF` wrapper for representing drifted distributions without duplicating drift logic across each PDF.

[Unreleased]: https://github.com/JuliaSpacePhysics/VelocityDistributionFunctions.jl/compare/v0.2.0...HEAD
