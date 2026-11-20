# VelocityDistributionFunctions

[![DOI](https://zenodo.org/badge/1072447322.svg)](https://doi.org/10.5281/zenodo.17905508)


Analysis tools for particle velocity distribution functions (VDFs).

**Installation**: at the Julia REPL, run `using Pkg; Pkg.add("VelocityDistributionFunctions")`

Start with the [MMS ion moments example](docs/src/moment_validation.md) to adapt a calibrated particle distribution.

## Conventions

- Angles are in degrees and give the particle velocity direction in physics spherical coordinates: `θ` is the polar angle from `ẑ` (0–180°), `φ` the azimuth from `x̂` (0–360°).

## API

Omit `tdim` for one scan. For a series, pass its time axis explicitly (e.g. `tdim = 1` for time-first data). Array outputs put time last. A shared `EnergyAngleGrid` applies to all scans; varying geometry takes a vector of grids. Supply `B` and `gyro_ref` as one three-component vector per scan. Other per-scan inputs are likewise vectors with one entry per scan, e.g. `eachcol(pa)` for pitch centers `pa[pitch, time]`.

| name | meaning |
|---|---|
| `angle_distribution(data, grid, coord; B, gyro_ref, bins = 12, method = :mean, tdim = nothing)` | per energy, the solid-angle-weighted mean (`method = :sum`: plain sum, for counts) of `data` over the cells whose center direction falls in each `coord` bin, as `(; data[bin, energy], solid_angle, edges, centers)`, `solid_angle` [sr] summing the finite cells of each bin. `coord` is `:pitch` (needs `B`), `:gyrophase` (needs `B` and `gyro_ref`; 0° along the perpendicular projection of `gyro_ref`, 90° along `B × gyro_ref`), `:phi` or `:theta`; `bins` is a count or edges [deg] |
| `omni_spectrum(data, grid; method = :mean, tdim = nothing)` | per energy, the solid-angle-weighted mean (`:sum`: plain sum) of `data` over all directions, as `[energy(, time)]`; a number for one scan without energies |
| `EnergyAngleGrid(energy, phi, theta; layout, denergy, dphi, dtheta)` | spherical velocity-space grid; `layout` names the data axes, e.g. `(:phi, :theta, :energy)`; `energy = nothing` for data without an energy axis (not for moments) |
| `plasma_moments(data, grid; species, units, sc_pot = 0, B = nothing, tdim = nothing)` | moments as a NamedTuple for one scan or a StructArray for a series; `sc_pot` may be shared or per scan. Supplying `B` adds field-aligned quantities |
| `convert_flux_units(data, energy, from => to; species = nothing)` | `data` converted between the `units` of `plasma_moments`; `energy` [eV] broadcasts against `data`; `species` is required between flux and phase-space density |
| `directional_spectra(S, pitch_angles, loss_cone; half_sector_width = nothing, para_tol = 0, perp_tol = 0, method = :mean, tdim = nothing)` | omni, parallel, antiparallel and perpendicular spectra `(; omni, para, anti, perp)` over the axes after pitch of a scan `S[pitch, rest...]`, e.g. energy spectra of `S[pitch, energy]`; a series accepts shared or per-scan pitch centers and loss cone; `method` is `:mean` (solid-angle-weighted mean; requires `half_sector_width`) or `:sum` (unweighted sum, for counts) |
| below | public, not exported |
| `pitch_angle_spectra(S, dE, min_channels, max_channels)` | for each channel range `min_channels[i]:max_channels[i]`, the `dE`-weighted mean of `S[pitch, energy, time]` over energy, skipping non-finite values, as `[pitch, time]` |

## Features

- [x] Pitch-angle, gyrophase, φ and θ distributions and omni energy spectra from 3D VDFs
- [x] Pitch-angle spectra averaged over energy channels
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