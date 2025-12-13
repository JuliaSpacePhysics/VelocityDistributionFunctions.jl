# Changelog

## Unreleased

### Changed

- **Breaking**: Drift velocity is no longer stored in the core PDF types (`MaxwellianPDF`, `KappaPDF`, `BiMaxwellianPDF`, `BiKappaPDF`).
  Drift is now handled by the public constructors via keyword arguments `u0` (e.g. `Kappa(...; u0=...)`).

- Distribution constructors support an `n` first argument by returning `VelocityDistribution(n, ...)`.

### Added

- `ShiftedPDF` wrapper for representing drifted distributions without duplicating drift logic across each PDF.
