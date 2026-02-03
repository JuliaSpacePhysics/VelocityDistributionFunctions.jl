"""
    Moments

Plasma moment calculations from particle distribution data.

# Moment Definitions

Given a velocity distribution function f(v), the moments are:

- **Density** (0th moment): n = ∫ f(v) d³v
- **Bulk velocity** (1st moment): V = (1/n) ∫ v f(v) d³v
- **Pressure tensor** (2nd moment): P_ij = m ∫ (v_i - V_i)(v_j - V_j) f(v) d³v
- **Heat flux** (3rd moment): Q_i = (m/2) ∫ (v_i - V_i)|v - V|² f(v) d³v

# Available Functions

- `density` - Number density
- `bulk_velocity` - Bulk flow velocity
- `pressure_tensor` - Full 3×3 pressure tensor
- `pressure_scalar` - Scalar pressure (trace of tensor / 3)
- `temperature_tensor` - Temperature tensor T = P / (n k_B)
- `temperature_scalar` - Scalar temperature
- `heat_flux` - Heat flux vector
- `entropy` - Entropy density

# Design Principles

1. Each moment function is independent and composable
2. Functions accept both `ParticleData` and analytical `AbstractVelocityPDF`
3. Full Unitful support when mass/charge are specified
4. Efficient numerical integration using solid angle weighting
"""

using LinearAlgebra: tr

# ============================================================================
# Density (0th moment)
# ============================================================================

"""
    density(pd::ParticleData)

Compute number density from particle distribution data.

The density is computed by integrating the distribution function over all velocity space:
    n = ∫ f(v) d³v

For discretized data in energy-angle bins:
    n = Σ_i f_i * v_i² * Δv_i * ΔΩ_i

where v_i is the velocity, Δv_i is the velocity bin width, and ΔΩ_i is the solid angle.

# Returns
- Scalar density for single-time data
- Vector `(nt,)` for time series data

# Notes
- Requires `mass` to be set in ParticleData for velocity conversion
- Input data should be in phase space density units (s³/m⁶ or s³/km⁶)
"""
function density(pd::ParticleData)
    pd.mass === nothing && throw(ArgumentError("ParticleData must have mass for moment calculations"))

    Ω = solid_angle(pd)
    v = [velocity_from_energy(E, pd.mass) for E in pd.energy]
    dv = _velocity_bin_widths(pd)

    if has_time(pd)
        result = zeros(promote_type(eltype(pd.data), eltype(v)), ntimes(pd))
        for t in 1:ntimes(pd)
            result[t] = _integrate_density(view(pd.data, :, :, :, t), v, dv, Ω)
        end
        return result
    else
        return _integrate_density(pd.data, v, dv, Ω)
    end
end

function _integrate_density(data, v, dv, Ω)
    n = zero(promote_type(eltype(data), eltype(v), eltype(dv), eltype(Ω)))
    for k in axes(data, 3)
        v_k = v[k]
        dv_k = dv[k]
        for j in axes(data, 2)
            for i in axes(data, 1)
                f = data[i, j, k]
                isfinite(f) || continue
                # d³v = v² dv dΩ
                n += f * v_k^2 * dv_k * Ω[i, j]
            end
        end
    end
    return n
end

# ============================================================================
# Bulk Velocity (1st moment)
# ============================================================================

"""
    bulk_velocity(pd::ParticleData)

Compute bulk flow velocity from particle distribution data.

The bulk velocity is the first moment of the distribution:
    V = (1/n) ∫ v f(v) d³v

# Returns
- Vector `[Vx, Vy, Vz]` for single-time data
- Matrix `(3, nt)` for time series data
"""
function bulk_velocity(pd::ParticleData)
    pd.mass === nothing && throw(ArgumentError("ParticleData must have mass for moment calculations"))

    Ω = solid_angle(pd)
    v_mag = [velocity_from_energy(E, pd.mass) for E in pd.energy]
    dv = _velocity_bin_widths(pd)
    vx, vy, vz = velocity_grid(pd)

    if has_time(pd)
        T = promote_type(eltype(pd.data), eltype(v_mag))
        result = zeros(T, 3, ntimes(pd))
        for t in 1:ntimes(pd)
            n, Vsum = _integrate_velocity(view(pd.data, :, :, :, t), vx, vy, vz, v_mag, dv, Ω)
            result[:, t] = n > 0 ? Vsum ./ n : fill(eltype(Vsum)(NaN), 3)
        end
        return result
    else
        n, Vsum = _integrate_velocity(pd.data, vx, vy, vz, v_mag, dv, Ω)
        return n > 0 ? Vsum ./ n : fill(eltype(Vsum)(NaN), 3)
    end
end

function _integrate_velocity(data, vx, vy, vz, v_mag, dv, Ω)
    T = promote_type(eltype(data), eltype(vx), eltype(v_mag), eltype(dv), eltype(Ω))
    n = zero(T)
    Vsum = zeros(T, 3)

    for k in axes(data, 3)
        v_k = v_mag[k]
        dv_k = dv[k]
        for j in axes(data, 2)
            for i in axes(data, 1)
                f = data[i, j, k]
                isfinite(f) || continue
                weight = f * v_k^2 * dv_k * Ω[i, j]
                n += weight
                Vsum[1] += weight * vx[i, j, k]
                Vsum[2] += weight * vy[i, j, k]
                Vsum[3] += weight * vz[i, j, k]
            end
        end
    end
    return n, Vsum
end

# ============================================================================
# Pressure Tensor (2nd moment)
# ============================================================================

"""
    pressure_tensor(pd::ParticleData; bulk_v=nothing)

Compute the pressure tensor from particle distribution data.

The pressure tensor is:
    P_ij = m ∫ (v_i - V_i)(v_j - V_j) f(v) d³v

# Arguments
- `pd`: ParticleData
- `bulk_v`: Optional pre-computed bulk velocity. If not provided, computed internally.

# Returns
- Symmetric 3×3 matrix for single-time data
- Vector of 3×3 matrices for time series data

# Notes
Returns pressure in units of [mass] × [velocity]² × [density] = [pressure]
"""
function pressure_tensor(pd::ParticleData; bulk_v=nothing)
    pd.mass === nothing && throw(ArgumentError("ParticleData must have mass for moment calculations"))

    Ω = solid_angle(pd)
    v_mag = [velocity_from_energy(E, pd.mass) for E in pd.energy]
    dv = _velocity_bin_widths(pd)
    vx, vy, vz = velocity_grid(pd)

    if has_time(pd)
        V = bulk_v === nothing ? bulk_velocity(pd) : bulk_v
        T = promote_type(eltype(pd.data), eltype(vx), typeof(pd.mass))
        result = [zeros(T, 3, 3) for _ in 1:ntimes(pd)]
        for t in 1:ntimes(pd)
            V_t = V[:, t]
            result[t] = _integrate_pressure(view(pd.data, :, :, :, t),
                                           vx, vy, vz, V_t, v_mag, dv, Ω, pd.mass)
        end
        return result
    else
        V = bulk_v === nothing ? bulk_velocity(pd) : bulk_v
        return _integrate_pressure(pd.data, vx, vy, vz, V, v_mag, dv, Ω, pd.mass)
    end
end

function _integrate_pressure(data, vx, vy, vz, V, v_mag, dv, Ω, mass)
    T = promote_type(eltype(data), eltype(vx), eltype(v_mag), typeof(mass))
    P = zeros(T, 3, 3)

    for k in axes(data, 3)
        v_k = v_mag[k]
        dv_k = dv[k]
        for j in axes(data, 2)
            for i in axes(data, 1)
                f = data[i, j, k]
                isfinite(f) || continue
                weight = mass * f * v_k^2 * dv_k * Ω[i, j]

                # Velocity relative to bulk
                dv_x = vx[i, j, k] - V[1]
                dv_y = vy[i, j, k] - V[2]
                dv_z = vz[i, j, k] - V[3]

                # Pressure tensor components
                P[1, 1] += weight * dv_x * dv_x
                P[1, 2] += weight * dv_x * dv_y
                P[1, 3] += weight * dv_x * dv_z
                P[2, 2] += weight * dv_y * dv_y
                P[2, 3] += weight * dv_y * dv_z
                P[3, 3] += weight * dv_z * dv_z
            end
        end
    end

    # Symmetrize
    P[2, 1] = P[1, 2]
    P[3, 1] = P[1, 3]
    P[3, 2] = P[2, 3]

    return Symmetric(P)
end

"""
    pressure_scalar(pd::ParticleData; bulk_v=nothing)

Compute scalar pressure (1/3 of the trace of pressure tensor).

P = (P_xx + P_yy + P_zz) / 3
"""
function pressure_scalar(pd::ParticleData; bulk_v=nothing)
    P = pressure_tensor(pd; bulk_v)
    if has_time(pd)
        return [tr(P_t) / 3 for P_t in P]
    else
        return tr(P) / 3
    end
end

# ============================================================================
# Temperature
# ============================================================================

"""
    temperature_tensor(pd::ParticleData; bulk_v=nothing, kB=1.380649e-23)

Compute temperature tensor from particle distribution data.

T_ij = P_ij / (n k_B)

# Arguments
- `pd`: ParticleData
- `bulk_v`: Optional pre-computed bulk velocity
- `kB`: Boltzmann constant (default: SI units)

# Returns
Temperature tensor in units of [energy] / [kB] = Kelvin
"""
function temperature_tensor(pd::ParticleData; bulk_v=nothing, kB=1.380649e-23)
    n = density(pd)
    P = pressure_tensor(pd; bulk_v)

    if has_time(pd)
        return [P[t] ./ (n[t] * kB) for t in eachindex(P)]
    else
        return P ./ (n * kB)
    end
end

"""
    temperature_scalar(pd::ParticleData; bulk_v=nothing, kB=1.380649e-23)

Compute scalar temperature.

T = P / (n k_B) = (1/3) Tr(T_tensor)
"""
function temperature_scalar(pd::ParticleData; bulk_v=nothing, kB=1.380649e-23)
    n = density(pd)
    P_scalar = pressure_scalar(pd; bulk_v)

    if has_time(pd)
        return P_scalar ./ (n .* kB)
    else
        return P_scalar / (n * kB)
    end
end

"""
    temperature_parallel(pd::ParticleData, B; bulk_v=nothing, kB=1.380649e-23)

Compute parallel temperature relative to magnetic field direction.

T_∥ = P_∥∥ / (n k_B)

where P_∥∥ is the pressure component along B.
"""
function temperature_parallel(pd::ParticleData, B; bulk_v=nothing, kB=1.380649e-23)
    n = density(pd)
    P = pressure_tensor(pd; bulk_v)

    if has_time(pd)
        if ndims(B) == 1
            b = normalize(SVector{3}(B))
            return [(b' * P[t] * b) / (n[t] * kB) for t in eachindex(P)]
        else
            return [(normalize(SVector{3}(B[:, t]))' * P[t] * normalize(SVector{3}(B[:, t]))) / (n[t] * kB)
                    for t in eachindex(P)]
        end
    else
        b = normalize(SVector{3}(B))
        return (b' * P * b) / (n * kB)
    end
end

"""
    temperature_perpendicular(pd::ParticleData, B; bulk_v=nothing, kB=1.380649e-23)

Compute perpendicular temperature relative to magnetic field direction.

T_⊥ = (P_total - P_∥∥) / (2 n k_B)

This is the average of the two perpendicular temperature components.
"""
function temperature_perpendicular(pd::ParticleData, B; bulk_v=nothing, kB=1.380649e-23)
    n = density(pd)
    P = pressure_tensor(pd; bulk_v)

    if has_time(pd)
        if ndims(B) == 1
            b = normalize(SVector{3}(B))
            P_para = [(b' * P[t] * b) for t in eachindex(P)]
            P_total = [tr(P[t]) for t in eachindex(P)]
            return (P_total .- P_para) ./ (2 .* n .* kB)
        else
            result = similar(n)
            for t in eachindex(P)
                b = normalize(SVector{3}(B[:, t]))
                P_para = b' * P[t] * b
                P_total = tr(P[t])
                result[t] = (P_total - P_para) / (2 * n[t] * kB)
            end
            return result
        end
    else
        b = normalize(SVector{3}(B))
        P_para = b' * P * b
        P_total = tr(P)
        return (P_total - P_para) / (2 * n * kB)
    end
end

# ============================================================================
# Heat Flux (3rd moment)
# ============================================================================

"""
    heat_flux(pd::ParticleData; bulk_v=nothing)

Compute heat flux vector from particle distribution data.

The heat flux is:
    Q_i = (m/2) ∫ (v_i - V_i)|v - V|² f(v) d³v

# Returns
- Vector `[Qx, Qy, Qz]` for single-time data
- Matrix `(3, nt)` for time series data

# Notes
Returns heat flux in units of [mass] × [velocity]³ × [density] = [power/area]
"""
function heat_flux(pd::ParticleData; bulk_v=nothing)
    pd.mass === nothing && throw(ArgumentError("ParticleData must have mass for moment calculations"))

    Ω = solid_angle(pd)
    v_mag = [velocity_from_energy(E, pd.mass) for E in pd.energy]
    dv = _velocity_bin_widths(pd)
    vx, vy, vz = velocity_grid(pd)

    if has_time(pd)
        V = bulk_v === nothing ? bulk_velocity(pd) : bulk_v
        T = promote_type(eltype(pd.data), eltype(vx), typeof(pd.mass))
        result = zeros(T, 3, ntimes(pd))
        for t in 1:ntimes(pd)
            V_t = V[:, t]
            result[:, t] = _integrate_heat_flux(view(pd.data, :, :, :, t),
                                                vx, vy, vz, V_t, v_mag, dv, Ω, pd.mass)
        end
        return result
    else
        V = bulk_v === nothing ? bulk_velocity(pd) : bulk_v
        return _integrate_heat_flux(pd.data, vx, vy, vz, V, v_mag, dv, Ω, pd.mass)
    end
end

function _integrate_heat_flux(data, vx, vy, vz, V, v_mag, dv, Ω, mass)
    T = promote_type(eltype(data), eltype(vx), eltype(v_mag), typeof(mass))
    Q = zeros(T, 3)

    for k in axes(data, 3)
        v_k = v_mag[k]
        dv_k = dv[k]
        for j in axes(data, 2)
            for i in axes(data, 1)
                f = data[i, j, k]
                isfinite(f) || continue

                # Velocity relative to bulk
                dv_x = vx[i, j, k] - V[1]
                dv_y = vy[i, j, k] - V[2]
                dv_z = vz[i, j, k] - V[3]
                dv_sq = dv_x^2 + dv_y^2 + dv_z^2

                weight = (mass / 2) * f * v_k^2 * dv_k * Ω[i, j] * dv_sq

                Q[1] += weight * dv_x
                Q[2] += weight * dv_y
                Q[3] += weight * dv_z
            end
        end
    end

    return Q
end

"""
    heat_flux_parallel(pd::ParticleData, B; bulk_v=nothing)

Compute heat flux component parallel to magnetic field.

Q_∥ = Q · b̂

where b̂ is the magnetic field unit vector.
"""
function heat_flux_parallel(pd::ParticleData, B; bulk_v=nothing)
    Q = heat_flux(pd; bulk_v)

    if has_time(pd)
        if ndims(B) == 1
            b = normalize(SVector{3}(B))
            return [dot(b, Q[:, t]) for t in axes(Q, 2)]
        else
            return [dot(normalize(SVector{3}(B[:, t])), Q[:, t]) for t in axes(Q, 2)]
        end
    else
        b = normalize(SVector{3}(B))
        return dot(b, Q)
    end
end

# ============================================================================
# Entropy
# ============================================================================

"""
    entropy_density(pd::ParticleData)

Compute entropy density from particle distribution data.

s = -k_B ∫ f ln(f) d³v

# Notes
This is the Boltzmann entropy. For numerical stability, uses f ln(f) → 0 as f → 0.
"""
function entropy_density(pd::ParticleData; kB=1.380649e-23)
    pd.mass === nothing && throw(ArgumentError("ParticleData must have mass for moment calculations"))

    Ω = solid_angle(pd)
    v_mag = [velocity_from_energy(E, pd.mass) for E in pd.energy]
    dv = _velocity_bin_widths(pd)

    if has_time(pd)
        T = promote_type(eltype(pd.data), eltype(v_mag))
        result = zeros(T, ntimes(pd))
        for t in 1:ntimes(pd)
            result[t] = _integrate_entropy(view(pd.data, :, :, :, t), v_mag, dv, Ω, kB)
        end
        return result
    else
        return _integrate_entropy(pd.data, v_mag, dv, Ω, kB)
    end
end

function _integrate_entropy(data, v_mag, dv, Ω, kB)
    T = promote_type(eltype(data), eltype(v_mag), typeof(kB))
    s = zero(T)

    for k in axes(data, 3)
        v_k = v_mag[k]
        dv_k = dv[k]
        for j in axes(data, 2)
            for i in axes(data, 1)
                f = data[i, j, k]
                (isfinite(f) && f > 0) || continue
                # s = -kB * f * ln(f)
                s -= kB * f * log(f) * v_k^2 * dv_k * Ω[i, j]
            end
        end
    end

    return s
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
Compute velocity bin widths from energy bins using dv = (dE/dv)^(-1) * dE = dE / (m*v).
"""
function _velocity_bin_widths(pd::ParticleData)
    v = [velocity_from_energy(E, pd.mass) for E in pd.energy]
    dE = pd.denergy

    # dv = dE / (m * v) = dE / sqrt(2 * m * E)
    dv = similar(v)
    for k in eachindex(v, dE)
        dv[k] = dE[k] / (pd.mass * v[k])
    end
    return dv
end

# ============================================================================
# Moments from Analytical VDFs
# ============================================================================

"""
    density(vd::VelocityDistribution)

Return the number density from an analytical velocity distribution.
"""
density(vd::VelocityDistribution) = vd.n

"""
    bulk_velocity(pdf::ShiftedPDF)

Return the bulk velocity from a shifted PDF.
"""
bulk_velocity(pdf::ShiftedPDF) = pdf.u0
bulk_velocity(::AbstractVelocityPDF) = SA[0.0, 0.0, 0.0]

"""
    bulk_velocity(vd::VelocityDistribution)

Return the bulk velocity from a velocity distribution.
"""
bulk_velocity(vd::VelocityDistribution) = bulk_velocity(vd.pdf)
