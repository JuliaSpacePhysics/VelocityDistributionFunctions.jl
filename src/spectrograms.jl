"""
    Spectrograms

Spectrogram computation functions for particle data.

# Available Functions

- `energy_spectrogram` - Flux vs energy (integrate over angles)
- `theta_spectrogram` - Flux vs polar angle
- `phi_spectrogram` - Flux vs azimuthal angle
- `pitch_angle_spectrogram` - Flux vs pitch angle
- `gyrophase_spectrogram` - Flux vs gyrophase angle

# Design Principles

Each function:
1. Takes `ParticleData` as primary input
2. Returns a named tuple with data and coordinates
3. Supports optional weighting (solid angle, etc.)
4. Can handle time series data naturally
"""

# ============================================================================
# Energy Spectrogram
# ============================================================================

"""
    energy_spectrogram(pd::ParticleData; weights=:solid_angle, method=:mean)

Compute energy spectrogram by integrating flux over all angles.

# Arguments
- `pd`: ParticleData
- `weights`: Weighting scheme
  - `:solid_angle` (default) - Weight by solid angle for proper averaging
  - `:uniform` - Uniform weighting
  - Custom array of shape `(nφ, nθ)`
- `method`: Aggregation method (`:mean` or `:sum`)

# Returns
Named tuple `(; data, energy)` where:
- `data`: Energy spectrogram array `(nE,)` or `(nE, nt)` for time series
- `energy`: Energy bin centers
"""
function energy_spectrogram(pd::ParticleData; weights=:solid_angle, method=:mean)
    _validate_method(method)

    w = _resolve_weights(pd, weights)
    T = promote_type(eltype(pd.data), eltype(w))

    if has_time(pd)
        result = zeros(T, nE(pd), ntimes(pd))
        counts = zeros(Int, nE(pd), ntimes(pd))
        for t in 1:ntimes(pd)
            _accumulate_energy!(view(result, :, t), view(counts, :, t),
                               view(pd.data, :, :, :, t), w, method)
        end
    else
        result = zeros(T, nE(pd))
        counts = zeros(Int, nE(pd))
        _accumulate_energy!(result, counts, pd.data, w, method)
    end

    _finalize_mean!(result, counts, method)
    return (; data=result, energy=pd.energy)
end

function _accumulate_energy!(result, counts, data, w, method)
    for k in axes(data, 3)
        for j in axes(data, 2)
            for i in axes(data, 1)
                val = data[i, j, k]
                if isfinite(val)
                    wt = method == :sum ? one(eltype(w)) : w[i, j]
                    result[k] += val * wt
                    counts[k] += 1
                end
            end
        end
    end
end

# ============================================================================
# Angular Spectrograms
# ============================================================================

"""
    theta_spectrogram(pd::ParticleData; energy_range=nothing, weights=:uniform, method=:mean)

Compute polar angle spectrogram by integrating over azimuthal angle and energy.

# Arguments
- `pd`: ParticleData
- `energy_range`: Optional `(Emin, Emax)` tuple to limit energy integration
- `weights`: Weighting scheme (`:uniform`, `:solid_angle`, or custom)
- `method`: Aggregation method (`:mean` or `:sum`)

# Returns
Named tuple `(; data, theta)` where:
- `data`: Theta spectrogram array `(nθ,)` or `(nθ, nt)`
- `theta`: Polar angle bin centers
"""
function theta_spectrogram(pd::ParticleData; energy_range=nothing, weights=:uniform, method=:mean)
    _validate_method(method)

    E_mask = _energy_mask(pd, energy_range)
    w = _resolve_weights(pd, weights)
    T = promote_type(eltype(pd.data), eltype(w))

    if has_time(pd)
        result = zeros(T, nθ(pd), ntimes(pd))
        counts = zeros(Int, nθ(pd), ntimes(pd))
        for t in 1:ntimes(pd)
            _accumulate_theta!(view(result, :, t), view(counts, :, t),
                              view(pd.data, :, :, :, t), w, E_mask, method)
        end
    else
        result = zeros(T, nθ(pd))
        counts = zeros(Int, nθ(pd))
        _accumulate_theta!(result, counts, pd.data, w, E_mask, method)
    end

    _finalize_mean!(result, counts, method)
    return (; data=result, theta=pd.theta)
end

function _accumulate_theta!(result, counts, data, w, E_mask, method)
    for k in axes(data, 3)
        E_mask[k] || continue
        for j in axes(data, 2)
            for i in axes(data, 1)
                val = data[i, j, k]
                if isfinite(val)
                    wt = method == :sum ? one(eltype(w)) : w[i, j]
                    result[j] += val * wt
                    counts[j] += 1
                end
            end
        end
    end
end

"""
    phi_spectrogram(pd::ParticleData; energy_range=nothing, weights=:uniform, method=:mean)

Compute azimuthal angle spectrogram by integrating over polar angle and energy.

# Arguments
- `pd`: ParticleData
- `energy_range`: Optional `(Emin, Emax)` tuple to limit energy integration
- `weights`: Weighting scheme
- `method`: Aggregation method (`:mean` or `:sum`)

# Returns
Named tuple `(; data, phi)` where:
- `data`: Phi spectrogram array `(nφ,)` or `(nφ, nt)`
- `phi`: Azimuthal angle bin centers
"""
function phi_spectrogram(pd::ParticleData; energy_range=nothing, weights=:uniform, method=:mean)
    _validate_method(method)

    E_mask = _energy_mask(pd, energy_range)
    w = _resolve_weights(pd, weights)
    T = promote_type(eltype(pd.data), eltype(w))

    if has_time(pd)
        result = zeros(T, nφ(pd), ntimes(pd))
        counts = zeros(Int, nφ(pd), ntimes(pd))
        for t in 1:ntimes(pd)
            _accumulate_phi!(view(result, :, t), view(counts, :, t),
                            view(pd.data, :, :, :, t), w, E_mask, method)
        end
    else
        result = zeros(T, nφ(pd))
        counts = zeros(Int, nφ(pd))
        _accumulate_phi!(result, counts, pd.data, w, E_mask, method)
    end

    _finalize_mean!(result, counts, method)
    return (; data=result, phi=pd.phi)
end

function _accumulate_phi!(result, counts, data, w, E_mask, method)
    for k in axes(data, 3)
        E_mask[k] || continue
        for j in axes(data, 2)
            for i in axes(data, 1)
                val = data[i, j, k]
                if isfinite(val)
                    wt = method == :sum ? one(eltype(w)) : w[i, j]
                    result[i] += val * wt
                    counts[i] += 1
                end
            end
        end
    end
end

# ============================================================================
# Pitch Angle Spectrogram
# ============================================================================

"""
    pitch_angle_spectrogram(pd::ParticleData, B; bins=nothing, energy_range=nothing, method=:mean)

Compute pitch angle spectrogram from particle data and magnetic field.

# Arguments
- `pd`: ParticleData
- `B`: Magnetic field vector `[Bx, By, Bz]` or time series `(3, nt)`
- `bins`: Pitch angle binning
  - `nothing` (default) - 15° bins from 0-180°
  - `Integer` - Number of uniform bins
  - `Vector` - Custom bin edges
- `energy_range`: Optional `(Emin, Emax)` tuple
- `method`: Aggregation method (`:mean` or `:sum`)

# Returns
Named tuple `(; data, pitch_angles, energy)` where:
- `data`: Pitch angle spectrogram `(nbins,)`, `(nbins, nE)`, `(nbins, nt)`, or `(nbins, nE, nt)`
- `pitch_angles`: Bin center angles in degrees
- `energy`: Energy bin centers (if energy dimension preserved)

# Examples
```julia
# Basic usage
result = pitch_angle_spectrogram(pd, B)

# With energy resolution
result = pitch_angle_spectrogram(pd, B; bins=18)  # 10° bins

# Limited energy range
result = pitch_angle_spectrogram(pd, B; energy_range=(100.0, 1000.0))
```
"""
function pitch_angle_spectrogram(pd::ParticleData, B; bins=nothing, energy_range=nothing, method=:mean)
    _validate_method(method)

    edges = _resolve_pitch_edges(bins)
    nbins = length(edges) - 1
    centers, _ = _compute_bin_properties(edges)
    E_mask = _energy_mask(pd, energy_range)
    nE_active = sum(E_mask)

    # Compute pitch angles for all angular bins
    pa = pitch_angles(pd, B)

    T = promote_type(eltype(pd.data), eltype(pa))

    if has_time(pd)
        nt = ntimes(pd)
        # For time series B, pa has shape (nφ, nθ, nt)
        pa_is_time_varying = ndims(pa) == 3

        if energy_range === nothing
            # Full energy resolution
            result = zeros(T, nbins, nE(pd), nt)
            counts = zeros(Int, nbins, nE(pd), nt)
            for t in 1:nt
                pa_t = pa_is_time_varying ? view(pa, :, :, t) : pa
                _accumulate_pa_energy!(view(result, :, :, t), view(counts, :, :, t),
                                       view(pd.data, :, :, :, t), pa_t, edges, E_mask, method)
            end
            _finalize_mean!(result, counts, method)
            return (; data=result, pitch_angles=centers, energy=pd.energy)
        else
            # Integrated over energy range
            result = zeros(T, nbins, nt)
            counts = zeros(Int, nbins, nt)
            for t in 1:nt
                pa_t = pa_is_time_varying ? view(pa, :, :, t) : pa
                _accumulate_pa!(view(result, :, t), view(counts, :, t),
                               view(pd.data, :, :, :, t), pa_t, edges, E_mask, method)
            end
            _finalize_mean!(result, counts, method)
            return (; data=result, pitch_angles=centers)
        end
    else
        if energy_range === nothing
            result = zeros(T, nbins, nE(pd))
            counts = zeros(Int, nbins, nE(pd))
            _accumulate_pa_energy!(result, counts, pd.data, pa, edges, E_mask, method)
            _finalize_mean!(result, counts, method)
            return (; data=result, pitch_angles=centers, energy=pd.energy)
        else
            result = zeros(T, nbins)
            counts = zeros(Int, nbins)
            _accumulate_pa!(result, counts, pd.data, pa, edges, E_mask, method)
            _finalize_mean!(result, counts, method)
            return (; data=result, pitch_angles=centers)
        end
    end
end

function _accumulate_pa!(result, counts, data, pa, edges, E_mask, method)
    n = length(edges)
    for k in axes(data, 3)
        E_mask[k] || continue
        for j in axes(data, 2)
            for i in axes(data, 1)
                pitch = pa[i, j]
                isfinite(pitch) || continue
                bin_idx = searchsortedfirst(edges, pitch) - 1
                bin_idx in (0, n) && continue

                val = data[i, j, k]
                if isfinite(val)
                    result[bin_idx] += val
                    counts[bin_idx] += 1
                end
            end
        end
    end
end

function _accumulate_pa_energy!(result, counts, data, pa, edges, E_mask, method)
    n = length(edges)
    for k in axes(data, 3)
        E_mask[k] || continue
        for j in axes(data, 2)
            for i in axes(data, 1)
                pitch = pa[i, j]
                isfinite(pitch) || continue
                bin_idx = searchsortedfirst(edges, pitch) - 1
                bin_idx in (0, n) && continue

                val = data[i, j, k]
                if isfinite(val)
                    result[bin_idx, k] += val
                    counts[bin_idx, k] += 1
                end
            end
        end
    end
end

# ============================================================================
# Gyrophase Spectrogram
# ============================================================================

"""
    gyrophase_spectrogram(pd::ParticleData, B; bins=nothing, energy_range=nothing, method=:mean)

Compute gyrophase spectrogram from particle data and magnetic field.

# Arguments
- `pd`: ParticleData
- `B`: Magnetic field vector `[Bx, By, Bz]` or time series `(3, nt)`
- `bins`: Gyrophase angle binning
  - `nothing` (default) - 15° bins from 0-360°
  - `Integer` - Number of uniform bins
  - `Vector` - Custom bin edges
- `energy_range`: Optional `(Emin, Emax)` tuple
- `method`: Aggregation method (`:mean` or `:sum`)

# Returns
Named tuple `(; data, gyrophase, energy)` where:
- `data`: Gyrophase spectrogram
- `gyrophase`: Bin center angles in degrees
- `energy`: Energy bin centers (if energy dimension preserved)
"""
function gyrophase_spectrogram(pd::ParticleData, B; bins=nothing, energy_range=nothing, method=:mean)
    _validate_method(method)

    edges = _resolve_gyro_edges(bins)
    nbins = length(edges) - 1
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    E_mask = _energy_mask(pd, energy_range)

    # Compute gyrophase angles for all angular bins
    gyro = gyrophase_angles(pd, B)

    T = promote_type(eltype(pd.data), eltype(gyro))

    if has_time(pd)
        nt = ntimes(pd)
        gyro_is_time_varying = ndims(gyro) == 3

        if energy_range === nothing
            result = zeros(T, nbins, nE(pd), nt)
            counts = zeros(Int, nbins, nE(pd), nt)
            for t in 1:nt
                gyro_t = gyro_is_time_varying ? view(gyro, :, :, t) : gyro
                _accumulate_gyro_energy!(view(result, :, :, t), view(counts, :, :, t),
                                         view(pd.data, :, :, :, t), gyro_t, edges, E_mask, method)
            end
            _finalize_mean!(result, counts, method)
            return (; data=result, gyrophase=centers, energy=pd.energy)
        else
            result = zeros(T, nbins, nt)
            counts = zeros(Int, nbins, nt)
            for t in 1:nt
                gyro_t = gyro_is_time_varying ? view(gyro, :, :, t) : gyro
                _accumulate_gyro!(view(result, :, t), view(counts, :, t),
                                 view(pd.data, :, :, :, t), gyro_t, edges, E_mask, method)
            end
            _finalize_mean!(result, counts, method)
            return (; data=result, gyrophase=centers)
        end
    else
        if energy_range === nothing
            result = zeros(T, nbins, nE(pd))
            counts = zeros(Int, nbins, nE(pd))
            _accumulate_gyro_energy!(result, counts, pd.data, gyro, edges, E_mask, method)
            _finalize_mean!(result, counts, method)
            return (; data=result, gyrophase=centers, energy=pd.energy)
        else
            result = zeros(T, nbins)
            counts = zeros(Int, nbins)
            _accumulate_gyro!(result, counts, pd.data, gyro, edges, E_mask, method)
            _finalize_mean!(result, counts, method)
            return (; data=result, gyrophase=centers)
        end
    end
end

function _accumulate_gyro!(result, counts, data, gyro, edges, E_mask, method)
    n = length(edges)
    for k in axes(data, 3)
        E_mask[k] || continue
        for j in axes(data, 2)
            for i in axes(data, 1)
                g = gyro[i, j]
                isfinite(g) || continue
                bin_idx = searchsortedfirst(edges, g) - 1
                # Handle wraparound for gyrophase
                bin_idx == 0 && (bin_idx = n - 1)
                bin_idx == n && (bin_idx = 1)

                val = data[i, j, k]
                if isfinite(val)
                    result[bin_idx] += val
                    counts[bin_idx] += 1
                end
            end
        end
    end
end

function _accumulate_gyro_energy!(result, counts, data, gyro, edges, E_mask, method)
    n = length(edges)
    for k in axes(data, 3)
        E_mask[k] || continue
        for j in axes(data, 2)
            for i in axes(data, 1)
                g = gyro[i, j]
                isfinite(g) || continue
                bin_idx = searchsortedfirst(edges, g) - 1
                bin_idx == 0 && (bin_idx = n - 1)
                bin_idx == n && (bin_idx = 1)

                val = data[i, j, k]
                if isfinite(val)
                    result[bin_idx, k] += val
                    counts[bin_idx, k] += 1
                end
            end
        end
    end
end

# ============================================================================
# Helper Functions
# ============================================================================

_resolve_gyro_edges(::Nothing) = collect(0.0:15.0:360.0)
_resolve_gyro_edges(n::Integer) = collect(range(0.0, 360.0; length=n+1))
function _resolve_gyro_edges(edges::AbstractVector)
    length(edges) >= 2 || throw(ArgumentError("edge vector must have ≥2 elements"))
    issorted(edges) || throw(ArgumentError("edge vector must be monotonically increasing"))
    first(edges) >= 0 && last(edges) <= 360 ||
        throw(ArgumentError("gyrophase edges must be in [0, 360]"))
    return edges
end

function _resolve_weights(pd::ParticleData, weights::Symbol)
    if weights == :solid_angle
        return solid_angle(pd)
    elseif weights == :uniform
        return ones(nφ(pd), nθ(pd))
    else
        throw(ArgumentError("Unknown weight type: $weights. Use :solid_angle, :uniform, or provide custom array."))
    end
end

function _resolve_weights(pd::ParticleData, weights::AbstractMatrix)
    size(weights) == (nφ(pd), nθ(pd)) || throw(DimensionMismatch("weights must have shape (nφ, nθ)"))
    return weights
end

function _energy_mask(pd::ParticleData, ::Nothing)
    return trues(nE(pd))
end

function _energy_mask(pd::ParticleData, energy_range::Tuple)
    Emin, Emax = energy_range
    return [Emin <= E <= Emax for E in pd.energy]
end

function _finalize_mean!(result, counts, method)
    method == :mean || return result
    T = eltype(result)
    for idx in eachindex(result, counts)
        result[idx] = iszero(counts[idx]) ? T(NaN) : result[idx] / counts[idx]
    end
    return result
end
