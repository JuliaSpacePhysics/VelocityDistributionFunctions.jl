# VelocityDistributionFunctions

[![DOI](https://zenodo.org/badge/1072447322.svg)](https://doi.org/10.5281/zenodo.17905508)
[![version](https://juliahub.com/docs/General/VelocityDistributionFunctions/stable/version.svg)](https://juliahub.com/ui/Packages/General/VelocityDistributionFunctions)

[![Build Status](https://github.com/JuliaSpacePhysics/VelocityDistributionFunctions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSpacePhysics/VelocityDistributionFunctions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSpacePhysics/VelocityDistributionFunctions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSpacePhysics/VelocityDistributionFunctions.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Analysis tools for particle velocity distribution functions (VDFs).

**Installation**: at the Julia REPL, run `using Pkg; Pkg.add("VelocityDistributionFunctions")`

**Documentation**: [![Dev](https://img.shields.io/badge/docs-dev-blue.svg?logo=julia)](https://JuliaSpacePhysics.github.io/VelocityDistributionFunctions.jl/dev/)

Some functions provided here are intended to be used as building blocks for mission-specific particle distribution tools. Mission-specific wrappers will generally be needed to load the particle data to be operated on, perform calibration, sanitization, and other preliminary steps.

## Conventions

- Angles are in degrees and give the particle velocity direction (not the instrument look direction) in physics spherical coordinates: `θ` is the polar angle from `ẑ` (0–180°), `φ` the azimuth from `x̂` (0–360°).
  For instruments reporting look directions, the velocity direction is `θ → 180 - θ`, `φ → φ + 180`.

## API

| name | meaning |
|---|---|
| `pitch_angle_distribution(vdf, B, φ, θ; bins = nothing, method = :mean)` | pitch-angle distribution of `vdf[φ, θ(, energy)]`, `bins` a count or edges (default 15° bins); returns `(; data, pitch_angles)` |
| `tpitch_angle_distribution(vdf, B, φ, θ; bins, method)` | the same with a trailing time dimension on `vdf`, `B`, and optionally `φ`, `θ` |
| `plasma_moments(dist, sc_pot = 0, magf = nothing; species, units = :eflux, edim = 1)` | moments of one distribution; field-aligned quantities when `magf` is given |
| `tmoments(data, theta, phi, energy, sc_pot = 0, magf = nothing; species, tdim = ndims(data), edim = 1, units)` | `plasma_moments` per time slice, as a `StructArray` |
| `directional_energy_spectra(S, pitch_angles, loss_cone; half_sector_width, para_tol = 0, perp_tol = 0)` | omni, parallel, antiparallel and perpendicular energy spectra `(; omni, para, anti, perp)` of `S[pitch, energy, time]` |
| below | public, not exported |
| `pitch_angle_spectra(S, dE, min_channels, max_channels)` | for each channel range `min_channels[i]:max_channels[i]`, the `dE`-weighted mean of `S[pitch, energy, time]` over energy, skipping non-finite values, as `[pitch, time]` |
| `sort_flux_by_pitch_angle!(flux, pitch_angle; sort_on_reverse = true)` | sort `flux[pitch, …, time]` and `pitch_angle[pitch, time]` into ascending pitch angle |
| `is_parallel(p, lc, tol)`, `is_anti_parallel`, `is_perpendicular` | whether pitch angle `p` lies within the loss cone `lc` about 0° or 180°, or between both cones, shrunk by `tol` [deg] |

## Features and Roadmap

- [ ] Fitting analytic distribution models (bi-Maxwellian, bi-kappa, …) to data
- [x] Generate pitch-angle distributions (PADs) from VDFs
    - [x] New PADs by averaging over multiple energy channels
- [x] Generate directional flux spectra (omni-, parallel, antiparallel, and perpendicular directions) from PADs
- [ ] Energy/gyro-phase spectra
- [x] Plasma moments (density, velocity, pressure, temperature, heat flux) from 3D distributions
- [ ] Extend to simulation data
- [ ] Volume visualization : scatter, volume, isosurface modes...
- [ ] Dispersion analysis (see also [PlasmaWaves.jl](https://github.com/JuliaSpacePhysics/PlasmaWaves.jl))
- [ ] Validation and Benchmarks

## Elsewhere

- [plasmapy.distributions](https://docs.plasmapy.org/en/stable/formulary/distribution.html)
- [ISEE_3D](https://earth-planets-space.springeropen.com/articles/10.1186/s40623-017-0761-9): Visualization tool for three-dimensional plasma velocity distributions
- [Bulk Flow Velocity Changing Software - Lynn B. Wilson](https://wind.nasa.gov/docs/vbulk_change_documentation.pdf)
- [SPEDAS Particle Tools Development Guide](https://spedas.org/presentations/pgs_development_v1.1.pdf)
- [PySPEDAS](https://pyspedas.readthedocs.io/en/latest/analysis.html#generalized-3-d-particle-distribution-tools)
    - [mms.mms_part_getspec](https://pyspedas.readthedocs.io/en/latest/mms_analysis.html#pyspedas.projects.mms.mms_part_getspec): Generate spectra and moments from 3D MMS particle data
    - [erg_analysis](https://pyspedas.readthedocs.io/en/latest/erg_analysis.html): Arase (ERG) Particle Tools
- [pyrfu](https://pyrfu.readthedocs.io/en/latest/index.html): Python package for working with Magnetospheric MultiScale (MMS) mission data
- [SpectralFitting.jl](https://juliaastro.org/SpectralFitting/stable/): Fast and flexible spectral fitting in Julia.
- [GDF](https://srijaniiserprinceton.github.io/GDF/index.html): gyrotorpic distribution functions, Slepian basis function based reconstructions to recover distribution functions