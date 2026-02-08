"""
    Spectrogram computation for particle distribution functions.

    Provides functions to compute energy, angular (theta, phi), pitch angle,
    and gyrophase spectrograms from particle data.

# Coordinate System Conventions

Follows SPEDAS/pyspedas conventions:
- **theta**: Polar/latitude angle in degrees, -90° (south) to +90° (north)
- **phi**: Azimuthal angle in degrees, 0° to 360° (typically from sun direction)
- **pitch angle**: Angle between particle velocity and magnetic field, 0° to 180°
- **gyrophase**: Angle in plane perpendicular to B field, 0° to 360°

# Data Structure

Input data should be a NamedTuple or struct with fields:
- `data`: Particle flux array `(phi, theta, energy)` or `(phi, theta, energy, time)`
- `theta`, `phi`: Angle arrays (can be 1-D or time-varying)
- `dtheta`, `dphi`: Angle bin widths (optional, estimated if missing)
- `energy`: Energy bin centers (can be 1-D or time-varying)
- `denergy`: Energy bin widths (optional)
- `bins`: Bin mask (1 = valid, 0 = masked/invalid) - optional
- For pitch/gyro: requires magnetic field `B`

# Units

Data are assumed to be in differential flux units (e.g., eV/(cm²·s·sr·eV) for energy flux).
The spectrograms preserve units but integrate over solid angle and/or energy as appropriate.
"""

# ============================================================================
# Energy Spectrograms
# ============================================================================

"""
    energy_spectrogram(data, theta, phi; weights=nothing, method=:mean)

Compute energy spectrogram by integrating over all angular directions.

# Arguments
- `data`: Particle flux array, either 3-D `(phi, theta, energy)` or 4-D `(phi, theta, energy, time)`
- `theta`: Polar/latitude angles in degrees, either 1-D or time-varying
- `phi`: Azimuthal angles in degrees, either 1-D or time-varying
- `weights`: Optional solid angle weights. If `nothing`, computed from theta/phi
- `method`: Aggregation method, either `:mean` (default), `:sum`, or `:median`

# Returns
Named tuple `(data, energy)` where:
- `data`: Energy spectrogram, 1-D `(energy,)` for 3-D input or 2-D `(energy, time)` for 4-D input
- `energy`: Energy bin indices (use with original energy array)

# Example
```julia
# For 3-D data (phi, theta, energy)
result = energy_spectrogram(flux, theta, phi)

# For 4-D time-series data (phi, theta, energy, time)
result = energy_spectrogram(flux, theta, phi)
plot(energy, result.data)  # energy vs time heatmap
```
"""
function energy_spectrogram(data, theta, phi; weights = nothing, method = :mean)
    _validate_method(method)
    sz = size(data)
    ndims(data) in (3, 4) || throw(ArgumentError("data must be 3-D or 4-D, got $(ndims(data))-D"))

    # Compute solid angle weights if not provided
    if weights === nothing
        weights = _compute_solid_angle_weights(theta, phi, sz)
    end

    # Integrate over angular dimensions
    if ndims(data) == 3
        spec = _integrate_angles(data, weights, method)
        return (; data = spec, energy = 1:sz[3])
    else  # 4-D with time
        n_energy, n_time = sz[3], sz[4]
        spec = similar(data, n_energy, n_time)
        for t in 1:n_time
            w = ndims(weights) == 3 ? weights[:, :, t] : weights
            spec[:, t] = _integrate_angles(view(data, :, :, :, t), w, method)
        end
        return (; data = spec, energy = 1:n_energy)
    end
end

# Helper to integrate over (phi, theta) dimensions
function _integrate_angles(data, weights, method)
    n_energy = size(data, 3)
    T = promote_type(eltype(data), eltype(weights))
    spec = zeros(T, n_energy)

    if method == :sum
        for e in 1:n_energy
            total = zero(T)
            for i in CartesianIndices(view(data, :, :, e))
                val = data[i, e]
                isfinite(val) && (total += val * weights[i])
            end
            spec[e] = total
        end
    elseif method == :mean
        for e in 1:n_energy
            total = zero(T)
            weight_sum = zero(T)
            for i in CartesianIndices(view(data, :, :, e))
                val = data[i, e]
                if isfinite(val)
                    w = weights[i]
                    total += val * w
                    weight_sum += w
                end
            end
            spec[e] = iszero(weight_sum) ? T(NaN) : total / weight_sum
        end
    else  # median
        spec = _integrate_angles_median(data, weights)
    end

    return spec
end

function _integrate_angles_median(data, weights)
    n_energy = size(data, 3)
    T = eltype(data)
    spec = zeros(T, n_energy)
    temp = Vector{T}()

    for e in 1:n_energy
        empty!(temp)
        for i in CartesianIndices(view(data, :, :, e))
            val = data[i, e]
            isfinite(val) && push!(temp, val)
        end
        spec[e] = isempty(temp) ? T(NaN) : median(temp)
    end

    return spec
end

# ============================================================================
# Angular Spectrograms (Theta and Phi)
# ============================================================================

"""
    theta_spectrogram(data, energy, phi; bins=nothing, weights=nothing, method=:mean)

Compute theta (polar/latitude angle) spectrogram by integrating over energy and phi.

# Arguments
- `data`: Particle flux array, 3-D `(phi, theta, energy)` or 4-D `(phi, theta, energy, time)`
- `energy`: Energy array or bin widths for weighting
- `phi`: Azimuthal angles in degrees
- `bins`: Theta binning (Integer for uniform bins, Vector for custom edges, or `nothing` for original)
- `weights`: Optional weights. If `nothing`, uses energy bin widths
- `method`: Aggregation method (`:mean`, `:sum`, or `:median`)

# Returns
Named tuple `(data, theta)` where:
- `data`: Theta spectrogram, 1-D `(n_theta,)` or 2-D `(n_theta, time)`
- `theta`: Theta bin centers in degrees

# Example
```julia
result = theta_spectrogram(flux, energy, phi; bins=18)  # 18 bins from -90° to 90°
```
"""
function theta_spectrogram(data, energy, phi; bins = nothing, weights = nothing, method = :mean)
    _validate_method(method)
    sz = size(data)
    ndims(data) in (3, 4) || throw(ArgumentError("data must be 3-D or 4-D, got $(ndims(data))-D"))

    n_theta = sz[2]

    # Handle binning
    if bins === nothing
        # No rebinning - use original theta dimension
        edges = range(-90, 90; length = n_theta + 1)
        bin_indices = 1:n_theta
    else
        edges = _resolve_angle_edges(bins, -90, 90)
        bin_indices = nothing  # Will bin on the fly
    end

    centers, _ = _compute_bin_properties(edges)

    # Compute weights
    if weights === nothing
        # Use energy bin widths as weights
        weights = _compute_energy_weights(energy, sz)
    end

    if ndims(data) == 3
        spec = _integrate_to_theta(data, weights, edges, bin_indices, method)
        return (; data = spec, theta = centers)
    else  # 4-D
        n_bins = length(centers)
        n_time = sz[4]
        spec = zeros(eltype(data), n_bins, n_time)
        for t in 1:n_time
            w = ndims(weights) == 4 ? view(weights, :, :, :, t) : weights
            spec[:, t] = _integrate_to_theta(view(data, :, :, :, t), w, edges, bin_indices, method)
        end
        return (; data = spec, theta = centers)
    end
end

"""
    phi_spectrogram(data, energy, theta; bins=nothing, weights=nothing, method=:mean)

Compute phi (azimuthal angle) spectrogram by integrating over energy and theta.

# Arguments
- `data`: Particle flux array, 3-D `(phi, theta, energy)` or 4-D `(phi, theta, energy, time)`
- `energy`: Energy array or bin widths for weighting
- `theta`: Polar angles in degrees
- `bins`: Phi binning (Integer for uniform bins, Vector for custom edges, or `nothing` for original)
- `weights`: Optional weights. If `nothing`, uses energy bin widths
- `method`: Aggregation method (`:mean`, `:sum`, or `:median`)

# Returns
Named tuple `(data, phi)` where:
- `data`: Phi spectrogram, 1-D `(n_phi,)` or 2-D `(n_phi, time)`
- `phi`: Phi bin centers in degrees

# Example
```julia
result = phi_spectrogram(flux, energy, theta; bins=32)  # 32 bins from 0° to 360°
```
"""
function phi_spectrogram(data, energy, theta; bins = nothing, weights = nothing, method = :mean)
    _validate_method(method)
    sz = size(data)
    ndims(data) in (3, 4) || throw(ArgumentError("data must be 3-D or 4-D, got $(ndims(data))-D"))

    n_phi = sz[1]

    # Handle binning
    if bins === nothing
        edges = range(0, 360; length = n_phi + 1)
        bin_indices = 1:n_phi
    else
        edges = _resolve_angle_edges(bins, 0, 360)
        bin_indices = nothing
    end

    centers, _ = _compute_bin_properties(edges)

    # Compute weights
    if weights === nothing
        weights = _compute_energy_weights(energy, sz)
    end

    if ndims(data) == 3
        spec = _integrate_to_phi(data, weights, edges, bin_indices, method)
        return (; data = spec, phi = centers)
    else  # 4-D
        n_bins = length(centers)
        n_time = sz[4]
        spec = zeros(eltype(data), n_bins, n_time)
        for t in 1:n_time
            w = ndims(weights) == 4 ? view(weights, :, :, :, t) : weights
            spec[:, t] = _integrate_to_phi(view(data, :, :, :, t), w, edges, bin_indices, method)
        end
        return (; data = spec, phi = centers)
    end
end

# ============================================================================
# Pitch Angle Spectrogram
# ============================================================================

"""
    pitch_angle_spectrogram(data, energy, theta, phi, B; bins=nothing, weights=nothing, method=:mean)

Compute pitch angle spectrogram by binning particles by their pitch angle and integrating over energy.

# Arguments
- `data`: Particle flux array, 3-D `(phi, theta, energy)` or 4-D `(phi, theta, energy, time)`
- `energy`: Energy array or bin widths
- `theta`: Polar angles in degrees
- `phi`: Azimuthal angles in degrees
- `B`: Magnetic field vector, either `(3,)` or `(3, time)` for time-varying
- `bins`: Pitch angle binning (Integer, Vector, or `nothing` for default 0:15:180)
- `weights`: Optional energy weights
- `method`: Aggregation method (`:mean`, `:sum`, or `:median`)

# Returns
Named tuple `(data, pitch_angles)` where:
- `data`: Pitch angle spectrogram, 1-D `(n_pitch,)` or 2-D `(n_pitch, time)`
- `pitch_angles`: Pitch angle bin centers in degrees

# Example
```julia
result = pitch_angle_spectrogram(flux, energy, theta, phi, B_field; bins=12)
```
"""
function pitch_angle_spectrogram(data, energy, theta, phi, B; bins = nothing, weights = nothing, method = :mean)
    _validate_method(method)
    sz = size(data)
    ndims(data) in (3, 4) || throw(ArgumentError("data must be 3-D or 4-D, got $(ndims(data))-D"))

    edges = _resolve_pitch_edges(bins)
    centers, _ = _compute_bin_properties(edges)
    n_bins = length(centers)

    # Compute energy weights
    if weights === nothing
        weights = _compute_energy_weights(energy, sz)
    end

    if ndims(data) == 3
        B_norm = normalize(SVector{3}(B))
        all(isfinite, B_norm) || return (; data = fill(NaN, n_bins), pitch_angles = centers)
        spec = _compute_pitch_spectrum(data, weights, theta, phi, B_norm, edges, method)
        return (; data = spec, pitch_angles = centers)
    else  # 4-D
        n_time = sz[4]
        spec = zeros(eltype(data), n_bins, n_time)
        B_reshaped = reshape(B, 3, :)

        for t in 1:n_time
            B_norm = normalize(SVector{3}(view(B_reshaped, :, t)))
            if all(isfinite, B_norm)
                w = ndims(weights) == 4 ? view(weights, :, :, :, t) : weights
                spec[:, t] = _compute_pitch_spectrum(view(data, :, :, :, t), w, theta, phi, B_norm, edges, method)
            else
                spec[:, t] .= NaN
            end
        end
        return (; data = spec, pitch_angles = centers)
    end
end

# ============================================================================
# Gyrophase Spectrogram
# ============================================================================

"""
    gyrophase_spectrogram(data, energy, theta, phi, B; bins=nothing, weights=nothing, method=:mean)

Compute gyrophase spectrogram by binning particles by their gyrophase angle and integrating over energy.

The gyrophase is the angle in the plane perpendicular to B, measuring rotation around the field.

# Arguments
- `data`: Particle flux array, 3-D `(phi, theta, energy)` or 4-D `(phi, theta, energy, time)`
- `energy`: Energy array or bin widths
- `theta`: Polar angles in degrees (typically in spacecraft coordinates)
- `phi`: Azimuthal angles in degrees
- `B`: Magnetic field vector, either `(3,)` or `(3, time)`
- `bins`: Gyrophase binning (Integer, Vector, or `nothing` for default 0:15:360)
- `weights`: Optional energy weights
- `method`: Aggregation method (`:mean`, `:sum`, or `:median`)

# Returns
Named tuple `(data, gyrophase)` where:
- `data`: Gyrophase spectrogram, 1-D `(n_gyro,)` or 2-D `(n_gyro, time)`
- `gyrophase`: Gyrophase bin centers in degrees

# Example
```julia
result = gyrophase_spectrogram(flux, energy, theta, phi, B_field; bins=24)
```
"""
function gyrophase_spectrogram(data, energy, theta, phi, B; bins = nothing, weights = nothing, method = :mean)
    _validate_method(method)
    sz = size(data)
    ndims(data) in (3, 4) || throw(ArgumentError("data must be 3-D or 4-D, got $(ndims(data))-D"))

    edges = _resolve_gyro_edges(bins)
    centers, _ = _compute_bin_properties(edges)
    n_bins = length(centers)

    # Compute energy weights
    if weights === nothing
        weights = _compute_energy_weights(energy, sz)
    end

    if ndims(data) == 3
        B_norm = normalize(SVector{3}(B))
        all(isfinite, B_norm) || return (; data = fill(NaN, n_bins), gyrophase = centers)
        spec = _compute_gyro_spectrum(data, weights, theta, phi, B_norm, edges, method)
        return (; data = spec, gyrophase = centers)
    else  # 4-D
        n_time = sz[4]
        spec = zeros(eltype(data), n_bins, n_time)
        B_reshaped = reshape(B, 3, :)

        for t in 1:n_time
            B_norm = normalize(SVector{3}(view(B_reshaped, :, t)))
            if all(isfinite, B_norm)
                w = ndims(weights) == 4 ? view(weights, :, :, :, t) : weights
                spec[:, t] = _compute_gyro_spectrum(view(data, :, :, :, t), w, theta, phi, B_norm, edges, method)
            else
                spec[:, t] .= NaN
            end
        end
        return (; data = spec, gyrophase = centers)
    end
end

# ============================================================================
# Helper Functions
# ============================================================================

# Validation
_validate_method(method) =
    method in (:mean, :sum, :median) || throw(ArgumentError("method must be :mean, :sum, or :median, got :$method"))

# Angle binning
_resolve_angle_edges(n::Integer, min_angle, max_angle) =
    range(min_angle, max_angle; length = Int(n) + 1)

function _resolve_angle_edges(edges::AbstractVector, min_angle, max_angle)
    length(edges) >= 2 || throw(ArgumentError("edge vector must have ≥2 elements"))
    issorted(edges) || throw(ArgumentError("edge vector must be monotonically increasing"))
    first(edges) >= min_angle && last(edges) <= max_angle ||
        throw(ArgumentError("edges must be in [$min_angle, $max_angle]"))
    return edges
end

# Reuse from pad.jl
_resolve_pitch_edges(::Nothing) = 0.0:15.0:180.0
_resolve_pitch_edges(n::Integer) = range(0.0, 180.0; length = Int(n) + 1)
_resolve_pitch_edges(edges::AbstractVector) = _resolve_angle_edges(edges, 0.0, 180.0)

_resolve_gyro_edges(::Nothing) = 0.0:15.0:360.0
_resolve_gyro_edges(n::Integer) = range(0.0, 360.0; length = Int(n) + 1)
_resolve_gyro_edges(edges::AbstractVector) = _resolve_angle_edges(edges, 0.0, 360.0)

function _compute_bin_properties(edges)
    centers = (edges[begin:(end - 1)] .+ edges[(begin + 1):end]) ./ 2
    half_widths = diff(collect(edges)) ./ 2
    return centers, half_widths
end

# Compute solid angle weights for energy spectrogram
function _compute_solid_angle_weights(theta, phi, sz)
    n_phi, n_theta = sz[1], sz[2]

    # Compute theta bin widths
    theta_1d = ndims(theta) == 1 ? theta : view(theta, :, 1)
    dtheta = _estimate_bin_widths(theta_1d)

    # Compute phi bin widths
    phi_1d = ndims(phi) == 1 ? phi : view(phi, :, 1)
    dphi = _estimate_bin_widths(phi_1d)

    # Solid angle: dΩ = sin(θ) dθ dφ (in radians)
    weights = zeros(n_phi, n_theta)
    for j in 1:n_theta
        for i in 1:n_phi
            θ = theta_1d[j]
            weights[i, j] = sind(θ) * deg2rad(dtheta[j]) * deg2rad(dphi[i])
        end
    end

    # Handle time-varying angles
    if ndims(theta) == 2 || ndims(phi) == 2
        n_time = max(size(theta, 2), size(phi, 2))
        weights_t = zeros(n_phi, n_theta, n_time)
        for t in 1:n_time
            theta_t = ndims(theta) == 2 ? view(theta, :, t) : theta_1d
            for j in 1:n_theta
                for i in 1:n_phi
                    weights_t[i, j, t] = weights[i, j]  # Could update with time-varying theta
                end
            end
        end
        return weights_t
    end

    return weights
end

# Compute energy weights
function _compute_energy_weights(energy, sz)
    n_phi, n_theta, n_energy = sz[1], sz[2], sz[3]
    n_time = length(sz) == 4 ? sz[4] : 1

    # Handle different energy formats
    if ndims(energy) == 1
        de = _estimate_bin_widths(energy)
        if n_time == 1
            weights = zeros(n_phi, n_theta, n_energy)
            for e in 1:n_energy
                weights[:, :, e] .= de[e]
            end
        else
            weights = zeros(n_phi, n_theta, n_energy, n_time)
            for t in 1:n_time, e in 1:n_energy
                weights[:, :, e, t] .= de[e]
            end
        end
    elseif size(energy) == (n_energy,)
        # Same as 1-D case
        de = _estimate_bin_widths(energy)
        weights = zeros(sz...)
        for idx in CartesianIndices(sz)
            e = idx[3]
            weights[idx] = de[e]
        end
    else
        # Assume energy is already the right size (possibly time-varying)
        weights = energy
    end

    return weights
end

# Estimate bin widths from centers
function _estimate_bin_widths(centers)
    n = length(centers)
    widths = similar(centers)

    if n == 1
        widths[1] = 1.0  # Arbitrary
    elseif n == 2
        widths[1] = widths[2] = abs(centers[2] - centers[1])
    else
        # Interior points
        for i in 2:(n - 1)
            widths[i] = abs(centers[i + 1] - centers[i - 1]) / 2
        end
        # Edges
        widths[1] = abs(centers[2] - centers[1])
        widths[n] = abs(centers[n] - centers[n - 1])
    end

    return widths
end

# Integrate to theta dimension
function _integrate_to_theta(data, weights, edges, bin_indices, method)
    n_phi, n_theta, n_energy = size(data)
    n_bins = length(edges) - 1
    T = promote_type(eltype(data), eltype(weights))

    if bin_indices !== nothing
        # No rebinning - simple integration
        spec = zeros(T, n_bins)
        for j in 1:n_theta
            total = zero(T)
            weight_sum = zero(T)
            count = 0
            for i in 1:n_phi, e in 1:n_energy
                val = data[i, j, e]
                if isfinite(val)
                    w = weights[i, j, e]
                    total += val * w
                    weight_sum += w
                    count += 1
                end
            end
            if method == :mean
                spec[j] = iszero(weight_sum) ? T(NaN) : total / weight_sum
            else
                spec[j] = total
            end
        end
    else
        # With rebinning (bins specified)
        # This would require actual theta values, skip for now
        error("Theta rebinning not yet implemented - use bins=nothing")
    end

    return spec
end

# Integrate to phi dimension
function _integrate_to_phi(data, weights, edges, bin_indices, method)
    n_phi, n_theta, n_energy = size(data)
    n_bins = length(edges) - 1
    T = promote_type(eltype(data), eltype(weights))

    if bin_indices !== nothing
        # No rebinning
        spec = zeros(T, n_bins)
        for i in 1:n_phi
            total = zero(T)
            weight_sum = zero(T)
            for j in 1:n_theta, e in 1:n_energy
                val = data[i, j, e]
                if isfinite(val)
                    w = weights[i, j, e]
                    total += val * w
                    weight_sum += w
                end
            end
            if method == :mean
                spec[i] = iszero(weight_sum) ? T(NaN) : total / weight_sum
            else
                spec[i] = total
            end
        end
    else
        error("Phi rebinning not yet implemented - use bins=nothing")
    end

    return spec
end

# Compute pitch angle spectrum
function _compute_pitch_spectrum(data, weights, theta, phi, B_norm, edges, method)
    n_phi, n_theta, n_energy = size(data)
    n_bins = length(edges) - 1
    T = promote_type(eltype(data), eltype(weights))

    spec = zeros(T, n_bins)
    counts = zeros(T, n_bins)

    # Get 1-D angle arrays
    theta_1d = ndims(theta) == 1 ? theta : view(theta, :, 1)
    phi_1d = ndims(phi) == 1 ? phi : view(phi, :, 1)

    for k in 1:n_theta
        sθ, cθ = sincosd(theta_1d[k])
        for j in 1:n_phi
            sφ, cφ = sincosd(phi_1d[j])
            # Look direction (assuming standard spacecraft coordinates)
            look = SA[-cφ * sθ, -sφ * sθ, -cθ]
            pitch = acosd(dot(look, B_norm))
            bin_idx = searchsortedfirst(edges, pitch) - 1
            bin_idx in (0, n_bins + 1) && continue

            # Accumulate over energy
            for e in 1:n_energy
                val = data[j, k, e]
                if isfinite(val)
                    w = weights[j, k, e]
                    spec[bin_idx] += val * w
                    counts[bin_idx] += w
                end
            end
        end
    end

    if method == :mean
        for i in 1:n_bins
            spec[i] = iszero(counts[i]) ? T(NaN) : spec[i] / counts[i]
        end
    end

    return spec
end

# Compute gyrophase spectrum
function _compute_gyro_spectrum(data, weights, theta, phi, B_norm, edges, method)
    n_phi, n_theta, n_energy = size(data)
    n_bins = length(edges) - 1
    T = promote_type(eltype(data), eltype(weights))

    spec = zeros(T, n_bins)
    counts = zeros(T, n_bins)

    # Get 1-D angle arrays
    theta_1d = ndims(theta) == 1 ? theta : view(theta, :, 1)
    phi_1d = ndims(phi) == 1 ? phi : view(phi, :, 1)

    # Compute reference vector perpendicular to B
    # Use the component of z perpendicular to B, or x if B || z
    z_axis = SA[0.0, 0.0, 1.0]
    perp_component = z_axis - dot(z_axis, B_norm) * B_norm

    if norm(perp_component) < 1e-6
        # B is nearly parallel to z, use x instead
        x_axis = SA[1.0, 0.0, 0.0]
        perp_component = x_axis - dot(x_axis, B_norm) * B_norm
    end

    ref_perp = normalize(perp_component)

    for k in 1:n_theta
        sθ, cθ = sincosd(theta_1d[k])
        for j in 1:n_phi
            sφ, cφ = sincosd(phi_1d[j])
            look = SA[-cφ * sθ, -sφ * sθ, -cθ]

            # Project look direction onto plane perpendicular to B
            look_perp = look - dot(look, B_norm) * B_norm
            look_perp_norm = norm(look_perp)

            # Skip if look is parallel to B
            look_perp_norm < 1e-6 && continue

            look_perp = look_perp / look_perp_norm

            # Compute gyrophase angle
            cos_gyro = dot(look_perp, ref_perp)
            # Use cross product to determine sign
            cross_prod = cross(ref_perp, look_perp)
            sin_gyro = dot(cross_prod, B_norm)
            gyro = atan(sin_gyro, cos_gyro) * 180 / π
            gyro < 0 && (gyro += 360)  # Map to [0, 360)

            bin_idx = searchsortedfirst(edges, gyro) - 1
            bin_idx in (0, n_bins + 1) && continue

            # Accumulate over energy
            for e in 1:n_energy
                val = data[j, k, e]
                if isfinite(val)
                    w = weights[j, k, e]
                    spec[bin_idx] += val * w
                    counts[bin_idx] += w
                end
            end
        end
    end

    if method == :mean
        for i in 1:n_bins
            spec[i] = iszero(counts[i]) ? T(NaN) : spec[i] / counts[i]
        end
    end

    return spec
end
