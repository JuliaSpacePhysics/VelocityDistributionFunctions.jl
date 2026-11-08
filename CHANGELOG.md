# Changelog

## [Unreleased]

### Removed

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
