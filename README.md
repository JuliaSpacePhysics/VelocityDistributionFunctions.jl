# VelocityDistributionFunctions

[![DOI](https://zenodo.org/badge/1072447322.svg)](https://doi.org/10.5281/zenodo.17905508)


Analysis tools for particle velocity distribution functions (VDFs).

**Installation**: at the Julia REPL, run `using Pkg; Pkg.add("VelocityDistributionFunctions")`

## Quickstart

```julia
using VelocityDistributionFunctions

energy = exp10.(range(1, 4; length = 32))  # eV
phi = 11.25:22.5:360                       # deg
theta = 11.25:22.5:180                     # deg
data = rand(32, 16, 8)                     # one scan: data[energy, phi, theta] in :eflux; any axis order, named by the grid
grid = EnergyAngleBins(:energy => energy, :phi => phi, :theta => theta)
B = [1.0, 0.0, 0.0]

pad = angle_distribution(data, grid, :pitch; B, bins = 18)                   # pad.data[pitch, energy]
spec = omni_spectrum(data, grid)                                             # spec[energy]
dirs = directional_spectra(pad.data, pad.centers, 30)                        # dirs.para[energy], ...
m = plasma_moments(data, grid; species = :e, units = :eflux, B)              # m.density, m.magt3, ...
psd = convert_flux_units(data, grid, :eflux => :df_cm; species = :e)

series = rand(32, 16, 8, 100)              # series[energy, phi, theta, time]
ms = plasma_moments(eachslice(series; dims = 4), grid; species = :e, units = :eflux)  # ms.density[time]
```

The [MMS ion moments example](docs/src/moment_validation.md) adapts instrument data with scan-varying tables and cross-validates against PySPEDAS.

## Conventions

- Angles are in degrees and give the particle velocity direction in physics spherical coordinates: `θ` is the polar angle from `ẑ` (0–180°), `φ` the azimuth from `x̂` (0–360°). Instrument look directions `(φ_look, θ_look)` convert as `φ = mod(φ_look + 180, 360)`, `θ = 180 - θ_look`.
- A series is a vector of scans, e.g. `eachslice(data; dims)` along the time axis `dims`. Per-scan outputs are stacked with time last; bin `edges` and `centers` are shared.
- In a series, other inputs are shared or given as a vector with one entry per scan: a vector of `EnergyAngleBins` for varying geometry, `eachcol(pa)` for pitch centers `pa[pitch, time]`. `B` and `gyro_ref` are always one three-component vector per scan.
- `method = :mean` is the solid-angle-weighted mean; `:sum` the plain sum, for counts.
- Non-finite values (e.g. `NaN` for dead or fill bins) are skipped; a bin, group or scan without finite values gives `NaN`. Set bins to `NaN` to exclude them, e.g. `data[energy .< 10, :, :] .= NaN` drops photoelectrons below 10 eV.
- `units` of `data`: `:eflux` [eV/(cm²·s·sr·eV)], `:flux` [1/(cm²·s·sr·eV)], `:df_cm` [s³/cm⁶], `:df_km` [s³/km⁶], `:df_m` [s³/m⁶].
- `species`: `:H`, `:He` (α), `:O` (¹⁶O⁺), `:e`, or a [PlasmaPopulations.jl](https://github.com/JuliaPlasma/PlasmaPopulations.jl) `Particle`.

## API

| name | meaning |
|---|---|
| `EnergyAngleBins(axes...; denergy, dphi, dtheta)` | energy [eV] and velocity-direction (`phi`, `theta` [deg]) bin centers and widths. Each argument names a data axis, in order: `name => values`, or `(names...) => (values...)` for coordinates sharing one axis (e.g. a flat list of directions). A vector lies along its axis; other arrays broadcast against the data. An angle without an axis spans its full range, e.g. gyrotropic `EnergyAngleBins(:theta => pitch, :energy => E)` with `B = [0, 0, 1]`. Cells span `θ ± dθ/2`, `φ ± dφ/2` within `θ ∈ [0, 180]`; uncovered directions count as empty. Missing widths are estimated from centers on an axis of their own. A bare `:energy` is an energy axis without values (not for moments) |
| `angle_distribution(data, grid, coord; B, gyro_ref, bins = 12, method = :mean)` | the mean of `data` over the cells whose center direction falls in each `coord` bin, as `(; data[bin(, energy)], solid_angle, edges, centers)`; `solid_angle` [sr] sums the finite cells of each bin. `coord` is `:pitch` (needs `B`), `:gyrophase` (needs `B` and `gyro_ref`; 0° along the perpendicular projection of `gyro_ref`, 90° along `B × gyro_ref`), `:phi` or `:theta`; `bins` is a count or edges [deg] |
| `omni_spectrum(data, grid; method = :mean)` | the mean of `data` over all directions, `[energy]` or a number without energies |
| `directional_spectra(S, pitch_angles, loss_cone; half_sector_width, para_tol = 0, perp_tol = 0, method = :mean)` | omni, parallel, antiparallel and perpendicular means `(; omni, para, anti, perp)` of `S[pitch, rest...]` over pitch, e.g. energy spectra; numbers for `S[pitch]`. `half_sector_width` [deg] is half the pitch width, estimated from the centers by default |
| `group_mean(data, groups; dims, weights = nothing)` | mean of `data` over each group along axis `dims`, weighted by `weights` (e.g. channel widths); `dims` becomes a group axis. `groups` are index ranges (e.g. `[1:3, 4:6]`) or `values => edges`, e.g. energy bands `E => [1e3, 1e4, 3e4]` with `E` per scan in a series |
| `plasma_moments(data, grid; species, units, sc_pot = 0, B = nothing)` | moments as a NamedTuple for one scan or a StructArray for a series, CGS with energies in eV, vectors and tensors in the frame of the angles: `density` [cm⁻³], `velocity`, `vthermal` [cm/s], `flux` [cm⁻² s⁻¹], `eflux` and plasma-frame heat flux `qflux` [eV cm⁻² s⁻¹], 3×3 `mftens`, `ptens` [eV cm⁻³] and `ttens` [eV], `avgtemp`, eigenvalues `t3` [eV]. `sc_pot` [V] gives energies at infinity `E + charge * sc_pot` (`charge` in e); bins pushed below zero drop out. `B` adds `magt3` [eV] along (`velocity` ⊥ `B`, `B × velocity`, `B`) and the temperature symmetry axis `symm` with `symm_theta`, `symm_phi`, `symm_ang` [deg]. Agyrotropy: `PlasmaPopulations.agyrotropy(m.ptens, B)` |
| `convert_flux_units(data, grid_or_energy, from => to; species = nothing)` | `data` converted between `units`, with energies from an `EnergyAngleBins` or an array [eV] that broadcasts against `data`; `species` is required between flux and phase-space density |

## Elsewhere

- [ISEE_3D](https://earth-planets-space.springeropen.com/articles/10.1186/s40623-017-0761-9): Visualization tool for three-dimensional plasma velocity distributions
- [Bulk Flow Velocity Changing Software - Lynn B. Wilson](https://wind.nasa.gov/docs/vbulk_change_documentation.pdf)
- [SPEDAS Particle Tools Development Guide](https://spedas.org/presentations/pgs_development_v1.1.pdf)
- [PySPEDAS](https://pyspedas.readthedocs.io/en/latest/analysis.html#generalized-3-d-particle-distribution-tools)
    - [mms.mms_part_getspec](https://pyspedas.readthedocs.io/en/latest/mms_analysis.html#pyspedas.projects.mms.mms_part_getspec): Generate spectra and moments from 3D MMS particle data
    - [erg_analysis](https://pyspedas.readthedocs.io/en/latest/erg_analysis.html): Arase (ERG) Particle Tools
- [pyrfu](https://pyrfu.readthedocs.io/en/latest/index.html): Python package for working with Magnetospheric MultiScale (MMS) mission data
- [GDF](https://srijaniiserprinceton.github.io/GDF/index.html): Slepian-basis reconstructions of gyrotropic distribution functions