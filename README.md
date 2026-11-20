# VelocityDistributionFunctions

[![DOI](https://zenodo.org/badge/1072447322.svg)](https://doi.org/10.5281/zenodo.17905508)


Analysis tools for particle velocity distribution functions (VDFs).

**Installation**: at the Julia REPL, run `using Pkg; Pkg.add("VelocityDistributionFunctions")`

Start with the [MMS ion moments example](docs/src/moment_validation.md) to adapt a calibrated particle distribution.

## Conventions

- Angles are in degrees and give the particle velocity direction in physics spherical coordinates: `θ` is the polar angle from `ẑ` (0–180°), `φ` the azimuth from `x̂` (0–360°).

## API

| name | meaning |
|---|---|
| `pitch_angle_distribution(vdf, B, φ, θ; bins = 12, method = :mean)` | pitch-angle distribution of `vdf[φ, θ(, energy)]`, `bins` a count or edges; returns `(; data, pitch_angles)` |
| `tpitch_angle_distribution(vdf, B, φ, θ; bins, method)` | the same with a trailing time dimension on `vdf`, `B`, and optionally on `φ`, `θ` |
| `EnergyAngleGrid(energy, phi, theta; layout, denergy, dphi, dtheta)` | spherical velocity-space grid; `layout` names the data axes, e.g. `(:phi, :theta, :energy)`; `energy = nothing` for data without an energy axis (not for moments) |
| `plasma_moments(data, grid; species, units, sc_pot = 0, B = nothing)` | moments of one distribution; field-aligned quantities when `B` is given |
| `tmoments(data, grid; species, units, sc_pot, B, tdim = ndims(data))` | `plasma_moments` per time slice, as a `StructArray`; vector arguments give one value per time |
| `convert_flux_units(data, energy, from => to; species = nothing)` | `data` converted between the `units` of `plasma_moments`; `energy` [eV] broadcasts against `data`; `species` is required between flux and phase-space density |
| `directional_energy_spectra(S, pitch_angles, loss_cone; half_sector_width = nothing, para_tol = 0, perp_tol = 0, method = :mean)` | omni, parallel, antiparallel and perpendicular energy spectra `(; omni, para, anti, perp)` of `S[pitch, energy, time]`; `method` is `:mean` (solid-angle-weighted mean; requires `half_sector_width`) or `:sum` (unweighted sum, for counts) |
| below | public, not exported |
| `pitch_angle_spectra(S, dE, min_channels, max_channels)` | for each channel range `min_channels[i]:max_channels[i]`, the `dE`-weighted mean of `S[pitch, energy, time]` over energy, skipping non-finite values, as `[pitch, time]` |

## Features

- [x] Pitch-angle distributions (PADs) from VDFs, and PADs averaged over energy channels
- [x] Directional (omni, parallel, antiparallel, perpendicular) energy spectra from PADs
- [ ] Energy/gyro-phase spectra
- [x] Plasma moments (density, velocity, pressure, temperature, heat flux) from 3D distributions, cross-validated against PySPEDAS ([example](docs/src/moment_validation.md))

## Elsewhere

- [ISEE_3D](https://earth-planets-space.springeropen.com/articles/10.1186/s40623-017-0761-9): Visualization tool for three-dimensional plasma velocity distributions
- [Bulk Flow Velocity Changing Software - Lynn B. Wilson](https://wind.nasa.gov/docs/vbulk_change_documentation.pdf)
- [SPEDAS Particle Tools Development Guide](https://spedas.org/presentations/pgs_development_v1.1.pdf)
- [PySPEDAS](https://pyspedas.readthedocs.io/en/latest/analysis.html#generalized-3-d-particle-distribution-tools)
    - [mms.mms_part_getspec](https://pyspedas.readthedocs.io/en/latest/mms_analysis.html#pyspedas.projects.mms.mms_part_getspec): Generate spectra and moments from 3D MMS particle data
    - [erg_analysis](https://pyspedas.readthedocs.io/en/latest/erg_analysis.html): Arase (ERG) Particle Tools
- [pyrfu](https://pyrfu.readthedocs.io/en/latest/index.html): Python package for working with Magnetospheric MultiScale (MMS) mission data
- [GDF](https://srijaniiserprinceton.github.io/GDF/index.html): Slepian-basis reconstructions of gyrotropic distribution functions