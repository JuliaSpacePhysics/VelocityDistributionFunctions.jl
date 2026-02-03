"""
    ParticleData

Particle measurement data structure for analyzing velocity distribution functions from instruments.

# Type Hierarchy

- `AbstractParticleData` - Base type for all particle data
- `ParticleData` - Concrete type for particle flux/counts in energy-angle bins

# Design Philosophy

This module provides a composable, modular interface for particle analysis:

1. **Separation of concerns**: Data structure is separate from analysis functions
2. **Single-responsibility functions**: Each function does one thing well
3. **Composability**: Functions can be chained and combined
4. **Type safety**: Full Unitful support and dimensional analysis
"""

# ============================================================================
# Abstract Types
# ============================================================================

"""
    AbstractParticleData

Base type for particle measurement data.
"""
abstract type AbstractParticleData end

Base.broadcastable(x::AbstractParticleData) = Ref(x)

# ============================================================================
# ParticleData Structure
# ============================================================================

"""
    ParticleData(data, energy, phi, theta; mass=me, charge=e)

Particle measurement data with energy-angle bins.

# Fields
- `data`: Particle flux/counts array with dimensions `(nφ, nθ, nE)` or `(nφ, nθ, nE, nt)` for time series
- `energy`: Energy bin centers (length `nE`)
- `phi`: Azimuthal angle centers in degrees (length `nφ`)
- `theta`: Polar angle centers in degrees (length `nθ`)
- `mass`: Particle mass (default: electron mass)
- `charge`: Particle charge (default: elementary charge)
- `denergy`: Energy bin widths (optional, computed from `energy` if not provided)
- `dphi`: Azimuthal bin widths in degrees (optional)
- `dtheta`: Polar bin widths in degrees (optional)

# Notes
- Angles are in spacecraft spherical coordinates
- `phi` ∈ [0, 360) is the azimuthal angle
- `theta` ∈ [0, 180] is the polar angle from the positive z-axis
- For velocity conversion: v = √(2E/m)

# Examples
```julia
# Basic construction
pd = ParticleData(flux, energies, phi_angles, theta_angles)

# With custom bin widths
pd = ParticleData(flux, energies, phi_angles, theta_angles;
                  denergy=energy_widths, dphi=phi_widths, dtheta=theta_widths)

# Ions vs electrons
pd_ions = ParticleData(flux, energies, phi, theta; mass=mp)
pd_electrons = ParticleData(flux, energies, phi, theta; mass=me)
```
"""
struct ParticleData{T, D<:AbstractArray{T}, E, A, M, C, DE, DA} <: AbstractParticleData
    data::D
    energy::E
    phi::A
    theta::A
    mass::M
    charge::C
    denergy::DE
    dphi::DA
    dtheta::DA

    function ParticleData(
        data::D, energy::E, phi::A1, theta::A2;
        mass::M=nothing, charge::C=nothing,
        denergy::DE=nothing, dphi::DA1=nothing, dtheta::DA2=nothing
    ) where {T, D<:AbstractArray{T}, E, A1, A2, M, C, DE, DA1, DA2}
        A = promote_type(A1, A2)
        DA = promote_type(DA1, DA2)

        # Validate dimensions
        ndims(data) in (3, 4) || throw(ArgumentError("data must be 3D (φ,θ,E) or 4D (φ,θ,E,t), got $(ndims(data))D"))
        size(data, 1) == length(phi) || throw(DimensionMismatch("phi length must match data dimension 1"))
        size(data, 2) == length(theta) || throw(DimensionMismatch("theta length must match data dimension 2"))
        size(data, 3) == length(energy) || throw(DimensionMismatch("energy length must match data dimension 3"))

        # Compute bin widths if not provided
        _denergy = denergy === nothing ? _compute_bin_widths(energy) : denergy
        _dphi = dphi === nothing ? _compute_bin_widths_periodic(phi, 360.0) : dphi
        _dtheta = dtheta === nothing ? _compute_bin_widths(theta) : dtheta

        return new{T, D, E, A, M, C, typeof(_denergy), typeof(_dphi)}(
            data, energy, phi, theta, mass, charge, _denergy, _dphi, _dtheta
        )
    end
end

# Convenience accessors
nφ(pd::ParticleData) = size(pd.data, 1)
nθ(pd::ParticleData) = size(pd.data, 2)
nE(pd::ParticleData) = size(pd.data, 3)
ntimes(pd::ParticleData) = ndims(pd.data) == 4 ? size(pd.data, 4) : 1
has_time(pd::ParticleData) = ndims(pd.data) == 4

# ============================================================================
# Bin Width Computation
# ============================================================================

"""
Compute bin widths assuming bins are contiguous.
For N centers, computes widths such that bins span from center[i] - width[i]/2 to center[i] + width[i]/2.
"""
function _compute_bin_widths(centers::AbstractVector)
    n = length(centers)
    n >= 2 || return fill(one(eltype(centers)), n)

    widths = similar(centers)
    # Interior bins: average of distances to neighbors
    for i in 2:(n-1)
        widths[i] = (centers[i+1] - centers[i-1]) / 2
    end
    # Edge bins: extrapolate from nearest interior
    widths[1] = centers[2] - centers[1]
    widths[n] = centers[n] - centers[n-1]
    return widths
end

"""
Compute bin widths for periodic coordinates (e.g., azimuthal angle).
"""
function _compute_bin_widths_periodic(centers::AbstractVector, period)
    n = length(centers)
    n >= 2 || return fill(period / n, n)

    widths = similar(centers)
    # All bins can use wrapped neighbors
    for i in eachindex(centers)
        i_prev = i == 1 ? n : i - 1
        i_next = i == n ? 1 : i + 1
        # Handle wraparound
        d_prev = centers[i] - centers[i_prev]
        d_next = centers[i_next] - centers[i]
        d_prev < 0 && (d_prev += period)
        d_next < 0 && (d_next += period)
        widths[i] = (d_prev + d_next) / 2
    end
    return widths
end

# ============================================================================
# Solid Angle Computation
# ============================================================================

"""
    solid_angle(pd::ParticleData)

Compute the solid angle element (steradians) for each angular bin.

Returns array of shape `(nφ, nθ)`.
"""
function solid_angle(pd::ParticleData)
    Ω = zeros(promote_type(eltype(pd.phi), eltype(pd.theta)), nφ(pd), nθ(pd))
    for (j, (θ, dθ)) in enumerate(zip(pd.theta, pd.dtheta))
        for (i, dφ) in enumerate(pd.dphi)
            # dΩ = sin(θ) dθ dφ
            Ω[i, j] = deg2rad(dφ) * abs(cosd(θ - dθ/2) - cosd(θ + dθ/2))
        end
    end
    return Ω
end

# ============================================================================
# Velocity Grid Computation
# ============================================================================

"""
    velocity_from_energy(E, mass)

Convert energy to velocity magnitude: v = √(2E/m).
"""
velocity_from_energy(E, mass) = sqrt(2 * E / mass)

"""
    velocity_grid(pd::ParticleData)

Compute 3D Cartesian velocity vectors for each bin center.

Returns tuple `(vx, vy, vz)` where each is an array of shape `(nφ, nθ, nE)`.
"""
function velocity_grid(pd::ParticleData)
    pd.mass === nothing && throw(ArgumentError("ParticleData must have mass defined for velocity conversion"))

    T = promote_type(eltype(pd.energy), eltype(pd.phi), typeof(pd.mass))
    vx = zeros(T, nφ(pd), nθ(pd), nE(pd))
    vy = zeros(T, nφ(pd), nθ(pd), nE(pd))
    vz = zeros(T, nφ(pd), nθ(pd), nE(pd))

    for (k, E) in enumerate(pd.energy)
        v = velocity_from_energy(E, pd.mass)
        for (j, θ) in enumerate(pd.theta)
            sθ, cθ = sincosd(θ)
            for (i, φ) in enumerate(pd.phi)
                sφ, cφ = sincosd(φ)
                vx[i, j, k] = v * sθ * cφ
                vy[i, j, k] = v * sθ * sφ
                vz[i, j, k] = v * cθ
            end
        end
    end
    return (vx, vy, vz)
end

# ============================================================================
# Look Direction Computation
# ============================================================================

"""
    look_directions(pd::ParticleData)

Compute unit look direction vectors for each angular bin.

Returns tuple `(lx, ly, lz)` where each is an array of shape `(nφ, nθ)`.

Note: Look direction points opposite to particle velocity (detector looks at incoming particles).
"""
function look_directions(pd::ParticleData)
    T = promote_type(eltype(pd.phi), eltype(pd.theta))
    lx = zeros(T, nφ(pd), nθ(pd))
    ly = zeros(T, nφ(pd), nθ(pd))
    lz = zeros(T, nφ(pd), nθ(pd))

    for (j, θ) in enumerate(pd.theta)
        sθ, cθ = sincosd(θ)
        for (i, φ) in enumerate(pd.phi)
            sφ, cφ = sincosd(φ)
            # Look direction is opposite to velocity direction
            lx[i, j] = -sθ * cφ
            ly[i, j] = -sθ * sφ
            lz[i, j] = -cθ
        end
    end
    return (lx, ly, lz)
end

# ============================================================================
# Coordinate Transformations
# ============================================================================

"""
    pitch_angles(pd::ParticleData, B)

Compute pitch angle (degrees) for each angular bin given magnetic field direction `B`.

# Arguments
- `pd`: ParticleData
- `B`: Magnetic field vector `[Bx, By, Bz]` or time series `(3, nt)`

Returns array of shape `(nφ, nθ)` or `(nφ, nθ, nt)` for time-varying B.
"""
function pitch_angles(pd::ParticleData, B::AbstractVector)
    length(B) == 3 || throw(ArgumentError("B must be 3-element vector"))

    B_norm = normalize(SVector{3}(B))
    all(isfinite, B_norm) || return fill(NaN, nφ(pd), nθ(pd))

    lx, ly, lz = look_directions(pd)
    pa = similar(lx)

    for j in axes(pa, 2), i in axes(pa, 1)
        look = SA[lx[i,j], ly[i,j], lz[i,j]]
        pa[i, j] = acosd(clamp(dot(look, B_norm), -1, 1))
    end
    return pa
end

function pitch_angles(pd::ParticleData, B::AbstractMatrix)
    size(B, 1) == 3 || throw(ArgumentError("B must have 3 rows"))
    nt = size(B, 2)

    lx, ly, lz = look_directions(pd)
    T = promote_type(eltype(lx), eltype(B))
    pa = zeros(T, nφ(pd), nθ(pd), nt)

    for t in 1:nt
        B_norm = normalize(SVector{3}(B[1,t], B[2,t], B[3,t]))
        if all(isfinite, B_norm)
            for j in axes(pa, 2), i in axes(pa, 1)
                look = SA[lx[i,j], ly[i,j], lz[i,j]]
                pa[i, j, t] = acosd(clamp(dot(look, B_norm), -1, 1))
            end
        else
            pa[:, :, t] .= NaN
        end
    end
    return pa
end

"""
    gyrophase_angles(pd::ParticleData, B, V=nothing)

Compute gyrophase angle (degrees) for each angular bin.

# Arguments
- `pd`: ParticleData
- `B`: Magnetic field vector `[Bx, By, Bz]` or time series `(3, nt)`
- `V`: Optional bulk velocity vector for gyrocenter frame correction

Returns array of shape `(nφ, nθ)` or `(nφ, nθ, nt)`.
"""
function gyrophase_angles(pd::ParticleData, B::AbstractVector, V=nothing)
    length(B) == 3 || throw(ArgumentError("B must be 3-element vector"))

    B_norm = normalize(SVector{3}(B))
    all(isfinite, B_norm) || return fill(NaN, nφ(pd), nθ(pd))

    # Build perpendicular basis
    e_perp = get_least_parallel_basis_vector(B_norm)
    e1 = normalize(cross(B_norm, e_perp))  # First perpendicular direction
    e2 = cross(B_norm, e1)                  # Second perpendicular direction

    lx, ly, lz = look_directions(pd)
    gyro = similar(lx)

    for j in axes(gyro, 2), i in axes(gyro, 1)
        look = SA[lx[i,j], ly[i,j], lz[i,j]]
        # Project onto perpendicular plane
        v1 = dot(look, e1)
        v2 = dot(look, e2)
        gyro[i, j] = atand(v2, v1)
    end
    # Convert to [0, 360) range
    gyro .= mod.(gyro, 360.0)
    return gyro
end

function gyrophase_angles(pd::ParticleData, B::AbstractMatrix, V=nothing)
    size(B, 1) == 3 || throw(ArgumentError("B must have 3 rows"))
    nt = size(B, 2)

    lx, ly, lz = look_directions(pd)
    T = promote_type(eltype(lx), eltype(B))
    gyro = zeros(T, nφ(pd), nθ(pd), nt)

    for t in 1:nt
        B_norm = normalize(SVector{3}(B[1,t], B[2,t], B[3,t]))
        if all(isfinite, B_norm)
            e_perp = get_least_parallel_basis_vector(B_norm)
            e1 = normalize(cross(B_norm, e_perp))
            e2 = cross(B_norm, e1)

            for j in axes(gyro, 2), i in axes(gyro, 1)
                look = SA[lx[i,j], ly[i,j], lz[i,j]]
                v1 = dot(look, e1)
                v2 = dot(look, e2)
                gyro[i, j, t] = mod(atand(v2, v1), 360.0)
            end
        else
            gyro[:, :, t] .= NaN
        end
    end
    return gyro
end
