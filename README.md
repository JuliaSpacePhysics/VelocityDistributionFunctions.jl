# VelocityDistributionFunctions

[![DOI](https://zenodo.org/badge/1072447322.svg)](https://doi.org/10.5281/zenodo.17905508)


Analysis tools for particle velocity distribution functions (VDFs).

**Installation**: at the Julia REPL, run `using Pkg; Pkg.add("VelocityDistributionFunctions")`

Start with the [MMS ion moments example](docs/src/moment_validation.md) to adapt a calibrated particle distribution.

## Conventions

- Angles are in degrees and give the particle velocity direction in physics spherical coordinates: `θ` is the polar angle from `ẑ` (0–180°), `φ` the azimuth from `x̂` (0–360°).

## API

Shared conventions:

- Omit `tdim` for one scan; for a series, pass its time axis (e.g. `tdim = 1` for time-first data). Outputs put time last.
- In a series, other inputs are shared or given as a vector with one entry per scan: a vector of `EnergyAngleBins` for varying geometry, `eachcol(pa)` for pitch centers `pa[pitch, time]`. `B` and `gyro_ref` are always one three-component vector per scan.
- A scan of pitch-resolved data has the pitch axis first.
- `method = :mean` is the solid-angle-weighted mean; `:sum` the plain sum, for counts.

| name | meaning |
|---|---|
| `EnergyAngleBins(axes...; denergy, dphi, dtheta)` | energy [eV] and velocity-direction (`phi`, `theta` [deg]) bin centers and widths. Each argument names a data axis, in order: `name => values`, or `(names...) => (values...)` for coordinates sharing one axis, e.g. `EnergyAngleBins(:energy => E, (:phi, :theta) => (φs, θs); dphi, dtheta)`. A vector lies along its axis; other arrays broadcast against the data. An angle without an axis spans its full range (gyrotropic or omnidirectional data). Missing widths are estimated from centers on an axis of their own. A bare `:energy` is an energy axis without values (not for moments) |
| `angle_distribution(data, grid, coord; B, gyro_ref, bins = 12, method = :mean, tdim = nothing)` | per energy, the mean of `data` over the cells whose center direction falls in each `coord` bin, as `(; data[bin, energy], solid_angle, edges, centers)`; `solid_angle` [sr] sums the finite cells of each bin. `coord` is `:pitch` (needs `B`), `:gyrophase` (needs `B` and `gyro_ref`; 0° along the perpendicular projection of `gyro_ref`, 90° along `B × gyro_ref`), `:phi` or `:theta`; `bins` is a count or edges [deg] |
| `omni_spectrum(data, grid; method = :mean, tdim = nothing)` | per energy, the mean of `data` over all directions, as `[energy(, time)]`; a number for one scan without energies |
| `directional_spectra(S, pitch_angles, loss_cone; half_sector_width = nothing, para_tol = 0, perp_tol = 0, method = :mean, tdim = nothing)` | omni, parallel, antiparallel and perpendicular means `(; omni, para, anti, perp)` of `S[pitch, rest...]` over pitch, e.g. energy spectra; numbers for `S[pitch]`. `:mean` needs `half_sector_width` |
| `group_mean(data, groups; dims, weights = nothing)` | mean of `data` over each group of indices along axis `dims` (e.g. energy channels `groups = [1:3, 4:6]`), weighted by `weights` (e.g. channel widths); `dims` becomes a group axis |
| `plasma_moments(data, grid; species, units, sc_pot = 0, B = nothing, tdim = nothing)` | moments as a NamedTuple for one scan or a StructArray for a series. Supplying `B` adds field-aligned quantities |
| `convert_flux_units(data, energy, from => to; species = nothing)` | `data` converted between the `units` of `plasma_moments`; `energy` [eV] broadcasts against `data`; `species` is required between flux and phase-space density |

## Features

- [x] Pitch-angle, gyrophase, φ and θ distributions and omni energy spectra from 3D VDFs
- [x] Weighted means over groups of channels, e.g. energy bands
- [x] Directional (omni, parallel, antiparallel, perpendicular) energy spectra from PADs
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