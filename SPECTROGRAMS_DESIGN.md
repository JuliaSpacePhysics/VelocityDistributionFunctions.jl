# Particle Spectrograms Implementation

## Overview

This document describes the implementation of particle data spectrogram functions in VelocityDistributionFunctions.jl, designed based on research of pyspedas/SPEDAS implementations while providing a cleaner, more intuitive API.

## Implemented Functions

### 1. Energy Spectrogram
```julia
energy_spectrogram(data, theta, phi; weights=nothing, method=:mean)
```
Integrates particle flux over all angular directions to produce energy vs time spectrogram.

**Key features:**
- Automatic solid angle weight computation if not provided
- Supports both 3-D `(phi, theta, energy)` and 4-D `(phi, theta, energy, time)` data
- Methods: `:mean` (default), `:sum`, `:median`

### 2. Angular Spectrograms
```julia
theta_spectrogram(data, energy, phi; bins=nothing, weights=nothing, method=:mean)
phi_spectrogram(data, energy, theta; bins=nothing, weights=nothing, method=:mean)
```
Produce theta or phi vs time spectrograms by integrating over energy and the other angle.

**Key features:**
- Optional rebinning (currently bins=nothing uses original binning)
- Energy-weighted integration
- Proper NaN handling

### 3. Pitch Angle Spectrogram
```julia
pitch_angle_spectrogram(data, energy, theta, phi, B; bins=nothing, weights=nothing, method=:mean)
```
Computes pitch angle distribution by binning particles based on angle between velocity and magnetic field.

**Key features:**
- Default binning: 0:15:180 degrees (12 bins)
- Handles time-varying magnetic field
- Energy-weighted integration over all energies in each pitch angle bin

### 4. Gyrophase Spectrogram
```julia
gyrophase_spectrogram(data, energy, theta, phi, B; bins=nothing, weights=nothing, method=:mean)
```
Computes gyrophase distribution (rotation angle in plane perpendicular to B).

**Key features:**
- Default binning: 0:15:360 degrees (24 bins)
- Computes reference perpendicular direction
- Handles particles nearly parallel to B gracefully

## Critical Implementation Details from SPEDAS Research

### From pyspedas Analysis

#### 1. **Solid Angle Weighting**
SPEDAS uses omega weights for proper solid angle integration. Key formula for simple solid angle:
```
dΩ = sin(θ) dθ dφ
```

More complex integrals (for moments) use:
- `ict = sin(θ₂) - sin(θ₁)` - integrated cos(theta) over bin
- `ic2t` - second moment in theta
- Various cross-terms for tensor components

**Our implementation**:
- Uses `omega_weights!` from `moments/omega_weights.jl` which exactly matches pyspedas
- For energy spectrogram, computes `dΩ = sin(θ) dθ dφ` per bin
- Properly normalizes by total solid angle when using `:mean` method

#### 2. **Bin Masking**
pyspedas explicitly handles invalid bins:
```python
zero_bins = np.argwhere(data['bins'] == 0)
if zero_bins.size != 0:
    for item in zero_bins:
        data['data'][item[0], item[1]] = 0.0
```

Then uses `nanmean` to ignore NaNs during averaging.

**Our implementation**:
- Accepts optional `bins` parameter in data structure
- Uses `isfinite()` checks to skip NaN/Inf values
- Properly normalizes by actual contributing weight, not total bins

#### 3. **Coordinate System**
SPEDAS uses:
- **Theta (latitude)**: -90° to +90° (or colatitude 0° to 180°)
- **Phi (azimuth)**: 0° to 360°
- **Phi wrapping**: Handles `phi_max == 0` → 360 and negative wrap-around

**Our implementation**:
- Documents clear coordinate conventions in module docstring
- Follows latitude convention (-90° to +90°)
- Phi from 0° to 360°
- Proper angle wrapping for gyrophase calculations

#### 4. **Angular Binning**
pyspedas theta/phi spectrograms handle four overlap cases:
1. Partial maximum overlap
2. Partial minimum overlap
3. Complete containment (data bin inside spec bin)
4. Complete coverage (spec bin inside data bin)

Weights by angular overlap: `(sin(θ_max) - sin(θ_min)) * Δφ`

**Our implementation**:
- Currently uses original binning (bins=nothing)
- Rebinning implementation marked as TODO for proper overlap handling
- Validates bin edges when custom bins provided

#### 5. **Energy Integration**
pyspedas integrates over energy using bin widths:
- If denergy not provided, estimates from bin centers
- Weights each energy channel by its width

**Our implementation**:
- `_estimate_bin_widths()` function for automatic width computation
- Uses energy widths as weights when integrating over energy
- Handles 1-D, multi-D, and time-varying energy arrays

#### 6. **Data Units**
SPEDAS supports: counts, rate, eflux, flux, df, df_cm, df_km, e2flux, e3flux

**Our implementation**:
- Assumes data in differential flux units
- Preserves input units through integration
- Documents unit assumptions in function docstrings

### From SPEDAS 3D Data Structures Wiki

#### Geometry Factor
The SPEDAS data structure includes `geom_factor` field (cm²·sr·eV/eV) representing:
```
gf = data.gf * data.eff
```

This is instrument-specific and accounts for detector effective area and efficiency.

**Our implementation**:
- Does not currently handle geometry factors (assumes pre-calibrated data)
- Could be added as optional calibration parameter

#### Solid Angle Integration
The moments calculation uses 13 omega weight channels for different tensor components.
We use the existing `omega_weights!` function that matches pyspedas exactly.

## Design Philosophy

### What Makes Our API Better

1. **Separation of Concerns**: Clear distinction between coordinate transformations and integration operations

2. **Flexible Inputs**: Accept both simple arrays and rich data structures

3. **Consistent Outputs**: All functions return `(data=..., bins=...)` named tuples

4. **Sensible Defaults**: Reasonable bin counts and methods without requiring expert knowledge

5. **Performance**: Leverage Julia's performance with proper typing and loop optimization

6. **Composability**: Functions work independently - users can mix and match

### What Could Be Improved

1. **Rebinning**: Currently disabled for theta/phi spectrograms - needs proper overlap handling

2. **Data Structure**: Could accept NamedTuple format matching `plasma_moments` for consistency

3. **Units Package**: Could integrate with Unitful.jl for automatic unit tracking

4. **Bin Quality Flags**: Could expose bin masking to users more explicitly

5. **Coordinate Transforms**: Could add field-aligned coordinate option like pyspedas `spd_pgs_do_fac`

6. **Optimization**: Could use Tullio.jl for tensor operations like the moments code does

## Usage Example

```julia
using VelocityDistributionFunctions

# Load particle data (phi, theta, energy, time)
data = ...  # Your particle flux data
theta = range(-90, 90, length=18)  # Latitude bins
phi = range(0, 360, length=32)  # Azimuth bins
energy = [10, 20, 50, 100, 200, 500, 1000]  # eV
B_field = ...  # Magnetic field (3, n_time)

# Energy spectrogram (omnidirectional flux)
omni = energy_spectrogram(data, theta, phi; method=:mean)
# Returns: (data = (n_energy, n_time), energy = 1:n_energy)

# Pitch angle spectrogram
pa = pitch_angle_spectrogram(data, energy, theta, phi, B_field; bins=12)
# Returns: (data = (12, n_time), pitch_angles = [7.5, 22.5, ...])

# Gyrophase spectrogram
gyro = gyrophase_spectrogram(data, energy, theta, phi, B_field; bins=24)
# Returns: (data = (24, n_time), gyrophase = [7.5, 22.5, ...])
```

## Testing Strategy

1. **Unit Tests**: Test each helper function independently
2. **Integration Tests**: Compare with pyspedas outputs for known datasets
3. **Edge Cases**: Test with NaN data, single time step, uniform distributions
4. **Performance**: Benchmark against equivalent pyspedas operations

## References

### pyspedas Source Code
- [Energy Spectrum](https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/spd_part_products/spd_pgs_make_e_spec.py)
- [Theta Spectrum](https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/spd_part_products/spd_pgs_make_theta_spec.py)
- [Phi/Gyrophase Spectrum](https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/spd_part_products/spd_pgs_make_phi_spec.py)
- [Omega Weights](https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d_omega_weights.py)

### Documentation
- [PySPEDAS Analysis Tools](https://pyspedas.readthedocs.io/en/latest/analysis.html)
- [SPEDAS 3D Data Structures](http://spedas.org/wiki/index.php?title=3D_data_structures)
- [MMS FPI Guide](https://spedas.org/wiki/index.php?title=MMS_Fast_Plasma_Instrument)
- [SPEDAS Paper](https://link.springer.com/article/10.1007/s11214-018-0576-4)

## Next Steps

1. Add proper bin masking support
2. Implement theta/phi rebinning with overlap handling
3. Add comprehensive tests comparing with pyspedas
4. Create example notebook with MMS data
5. Consider adding field-aligned coordinate option
6. Performance optimization with Tullio.jl
