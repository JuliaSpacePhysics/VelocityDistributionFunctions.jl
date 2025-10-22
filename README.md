# VelocityDistributionFunctions

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSpacePhysics.github.io/VelocityDistributionFunctions.jl/dev/)
[![Build Status](https://github.com/JuliaSpacePhysics/VelocityDistributionFunctions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaSpacePhysics/VelocityDistributionFunctions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaSpacePhysics/VelocityDistributionFunctions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSpacePhysics/VelocityDistributionFunctions.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Analysis tools for particle velocity distribution functions (VDFs).

Functions provided here are intended to be used as building blocks for mission-specific particle distribution tools. Mission-specific wrappers will generally be needed to load the particle data to be operated on, perform calibration, sanitization, and other preliminary steps.

## Features and Roadmap

- [x] Generate pitch-angle distributions (PADs) from VDFs
    - [x] New PADs by averaging over multiple energy channels
- [x] Generate directional flux spectra (omni-, parallel, antiparallel, and perpendicular directions) from PADs
- [ ] Energy/gyro-phase spectra
- [ ] Moments
- [ ] Extend to simulation data
- [ ] Validation and Benchmarks

## Elsewhere

- [ISEE_3D](https://earth-planets-space.springeropen.com/articles/10.1186/s40623-017-0761-9): Visualization tool for three-dimensional plasma velocity distributions
- [Bulk Flow Velocity Changing Software - Lynn B. Wilson](https://wind.nasa.gov/docs/vbulk_change_documentation.pdf)
- [SPEDAS Particle Tools Development Guide](https://spedas.org/presentations/pgs_development_v1.1.pdf)
- [PySPEDAS](https://pyspedas.readthedocs.io/en/latest/analysis.html#generalized-3-d-particle-distribution-tools)
    - [mms.mms_part_getspec](https://pyspedas.readthedocs.io/en/latest/mms_analysis.html#pyspedas.projects.mms.mms_part_getspec): Generate spectra and moments from 3D MMS particle data
    - [erg_analysis](https://pyspedas.readthedocs.io/en/latest/erg_analysis.html): Arase (ERG) Particle Tools
- [pyrfu](https://pyrfu.readthedocs.io/en/latest/index.html): Python package for working with Magnetospheric MultiScale (MMS) mission data