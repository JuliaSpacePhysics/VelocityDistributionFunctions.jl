"""
    pitch_angle_distribution(vdf, B, φ, θ; bins=nothing, method=:mean) -> (; data, pitch_angles)

Compute pitch-angle distributions from particle `vdf` and magnetic field `B` measurements, given azimuthal `φ` and polar `θ` angles.

Returns a named tuple with fields:
- `data`: Pitch-angle distribution array `(nbins,)` for 2-D input `vdf` or `(nbins, energy)` for 3-D input `vdf`
- `pitch_angles`: Bin center angles in degrees

# Notes
- `vdf` can be either 2-D `(φ, θ)` or 3-D `(φ, θ, energy)`
- `bins` is used for pitch angle binning. It can be an `Integer` for uniform bin count, or a `Vector` of bin edges.
- `method` is used for aggregation. It can be `:mean` (default) or `:sum`.
"""
function pitch_angle_distribution(vdf, B, φ, θ; bins = nothing, method = :mean)
    _validate_method(method)

    sz = size(vdf)
    _validate_angle_array(φ, sz[1])
    _validate_angle_array(θ, sz[2])

    edges = _resolve_pitch_edges(bins)
    nbins = length(edges) - 1
    centers, _ = _compute_bin_properties(edges)

    pad_size = ndims(vdf) == 2 ? (nbins,) : (nbins, sz[3])
    counts_size = ndims(vdf) == 2 ? nbins : (nbins, sz[3])
    pad = zeros(eltype(vdf), pad_size)
    counts = zeros(Int, counts_size)

    _accumulate_pad!(pad, counts, vdf, B, φ, θ, edges; method)

    return (; data = pad, pitch_angles = centers)
end

@inline tslice(x) = eachslice(x; dims = ndims(x))

"""
    tpitch_angle_distribution(vdf, B, φ, θ; bins=nothing, method=:mean)

Compute pitch-angle distributions ***time series*** from particle `vdf` and magnetic field `B` measurements, given azimuthal `φ` and polar `θ` angles.

Returns a named tuple with fields:
- `data`: Pitch-angle distribution array, `(nbins, time)` for 3-D or `(nbins, energy, time)` for 4-D input `vdf`
- `pitch_angles`: Bin center angles in degrees

# Notes
- `vdf`: can be either 3-D `(φ, θ, time)` or 4-D `(φ, θ, energy, time)`. `B` should match the time dimension of `vdf`
- `φ` and `θ` can be either 1-D `(n,)` or time-varying 2-D `(n, time)`
- `bins` is used for pitch angle binning. It can be an `Integer` for uniform bin count, or a `Vector` of bin edges.
- `method` is used for aggregation. It can be `:mean` (default) or `:sum`.
"""
function tpitch_angle_distribution(vdf, B, φ, θ; bins = nothing, method = :mean)
    _validate_method(method)
    sz = size(vdf)
    _validate_angle_array(φ, sz[1], sz[end])
    _validate_angle_array(θ, sz[2], sz[end])

    edges = _resolve_pitch_edges(bins)
    nbins = length(edges) - 1
    centers, _ = _compute_bin_properties(edges)

    pad_size = ndims(vdf) == 3 ? (nbins, sz[end]) : (nbins, sz[3], sz[end])
    counts_size = ndims(vdf) == 3 ? nbins : (nbins, sz[3])
    pad = zeros(eltype(vdf), pad_size)
    counts = zeros(Int, counts_size)
    # reshape to handle vector input (time-invariant)
    B_slices = tslice(reshape(B, 3, :))
    φ_slices = tslice(reshape(φ, sz[1], :))
    θ_slices = tslice(reshape(θ, sz[2], :))
    _accumulate_pad!.(tslice(pad), (counts,), tslice(vdf), B_slices, φ_slices, θ_slices, (edges,); method)

    return (; data = pad, pitch_angles = centers)
end

# Validation
_validate_method(method) =
    method in (:mean, :sum) || throw(ArgumentError("method must be :mean or :sum, got :$method"))

function _validate_angle_array(angles, n, n_time = 1)
    @assert ndims(angles) in (1, 2)
    @assert size(angles, 1) == n
    @assert size(angles, 2) in (1, n_time)
    return
end

# Pitch angle binning
_resolve_pitch_edges(::Nothing) = 0.0:15.0:180.0
_resolve_pitch_edges(n) = range(0.0, 180.0; length = Int(n) + 1)

function _resolve_pitch_edges(edges::AbstractVector)
    length(edges) >= 2 || throw(ArgumentError("edge vector must have ≥2 elements, got $(length(edges))"))
    issorted(edges) || throw(ArgumentError("edge vector must be monotonically increasing"))
    first(edges) >= 0 && last(edges) <= 180 ||
        throw(ArgumentError("edges must be in [0, 180], got [$(first(edges)), $(last(edges))]"))
    return edges
end

function _compute_bin_properties(edges)
    centers = (edges[begin:(end - 1)] .+ edges[(begin + 1):end]) ./ 2
    half_widths = diff(collect(edges)) ./ 2
    return centers, half_widths
end

# Core accumulation: works for both 2-D (φ, θ) and 3-D (φ, θ, energy)
function _accumulate_pad!(pad, counts, vdf, B, φ, θ, edges; method = :mean)
    counts .= 0
    B_norm = normalize(SVector{3}(B))
    all(isfinite, B_norm) || return fill!(pad, NaN)
    n = length(edges)

    for (k, θ_k) in enumerate(θ)
        sθ, cθ = sincosd(θ_k)
        for (j, φ_j) in enumerate(φ)
            sφ, cφ = sincosd(φ_j)
            look = SA[-cφ * sθ, -sφ * sθ, -cθ]
            pitch = acosd(dot(look, B_norm))
            bin_idx = searchsortedfirst(edges, pitch) - 1
            bin_idx in (0, n) && continue
            # Accumulate across energy dimension
            for e in axes(vdf, 3)
                value = vdf[j, k, e]
                if isfinite(value)
                    pad[bin_idx, e] += value
                    counts[bin_idx, e] += 1
                end
            end
        end
    end
    method == :mean && begin
        T = eltype(pad)
        for idx in eachindex(pad, counts)
            pad[idx] = iszero(counts[idx]) ? T(NaN) : pad[idx] / counts[idx]
        end
    end
    return pad
end
