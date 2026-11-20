para_losscone(l) = min(l, 180 - l)
is_parallel(p, l, atol) = p < (para_losscone(l) - atol)
is_anti_parallel(p, l, atol) = p > (180.0 + atol - para_losscone(l))
is_perpendicular(p, l, atol) = (pl = para_losscone(l); (pl + atol) < p < (180.0 - pl - atol))


"""
    directional_spectra(S, pitch_angles, loss_cone; half_sector_width = nothing, para_tol = 0, perp_tol = 0, method = :mean, tdim = nothing)

With `l = min(loss_cone, 180° - loss_cone)`, sectors at pitch angle `θ` are
para if `θ < l - para_tol`, anti if `θ > 180° - l + para_tol`, perp if `l + perp_tol < θ < 180° - l - perp_tol`; omni takes all.
Sectors with non-finite flux or pitch angle are skipped; a direction left without sectors gives `NaN`.

`:mean` weights sectors by solid angle `2Δθ sin θ`, or `Δθ sin Δθ` within `Δθ = half_sector_width` of 0° or 180°.
`:sum` does not compensate for skipped sectors.
"""
function directional_spectra(S, pitch_angles, loss_cone; half_sector_width = nothing, para_tol = 0, perp_tol = 0, method = :mean, tdim = nothing)
    method in (:mean, :sum) || throw(ArgumentError("method must be :mean or :sum, got :$method"))
    weighted = method === :mean
    weighted && isnothing(half_sector_width) && throw(ArgumentError("method = :mean requires half_sector_width"))
    ndims(S) > !isnothing(tdim) || throw(DimensionMismatch("S needs a pitch axis besides tdim"))
    Base.require_one_based_indexing(S)
    shared = pitch_angles isa AbstractVector{<:Number}
    T = float(promote_type(eltype(S), shared ? eltype(pitch_angles) : eltype(eltype(pitch_angles)), eltype(loss_cone)))
    scan_size = isnothing(tdim) ? size(S) : ntuple(i -> size(S, i + (i >= tdim)), ndims(S) - 1)
    n_pa, spectrum_size = first(scan_size), Base.tail(scan_size)
    weight = _sector_weight(T, weighted ? half_sector_width : 0, weighted)
    buf = (; dΩ = Vector{T}(undef, n_pa), para = Vector{T}(undef, n_pa), anti = Vector{T}(undef, n_pa), perp = Vector{T}(undef, n_pa))
    scan!(out, t, s, centers, lc) = _directional_scan!(out, t, buf, s, centers, lc, weight, para_tol, perp_tol, weighted)
    n_t = isnothing(tdim) ? 1 : _scan_count(S, tdim)
    _check_pitch_angles(pitch_angles, n_pa, n_t, !isnothing(tdim))
    if isnothing(tdim)
        loss_cone isa Number || throw(DimensionMismatch("one scan takes a scalar loss cone"))
        out = ntuple(_ -> Array{T}(undef, spectrum_size), 4)
        scan!(out, 1, S, pitch_angles, loss_cone)
        return _spectra_tuple(isempty(spectrum_size) ? map(only, out) : out)
    end
    loss_cone isa Number || _check_series(loss_cone, n_t, "loss_cone")
    outs = ntuple(_ -> Array{T}(undef, spectrum_size..., n_t), 4)
    _foreach_scan(eachslice(S; dims = tdim)) do t, s
        scan!(outs, t, s, shared ? pitch_angles : pitch_angles[t], _at(loss_cone, t))
    end
    return _spectra_tuple(outs)
end

_spectra_tuple(out) = NamedTuple{(:omni, :para, :anti, :perp)}(out)
# Function barrier: the slice type depends on the runtime `tdim`.
_foreach_scan(f, scans) = foreach(f, eachindex(scans), scans)

# Pitch centers are one vector of `n_pa` angles shared by all scans or, in a series, one such vector per scan.
function _check_pitch_angles(pitch_angles, n_pa, n_t, series)
    per_scan = series && !(pitch_angles isa AbstractVector{<:Number})
    per_scan && _check_series(pitch_angles, n_t, "pitch_angles")
    all(c -> axes(c, 1) == Base.OneTo(n_pa), per_scan ? pitch_angles : (pitch_angles,)) ||
        throw(DimensionMismatch("pitch_angles needs $n_pa centers per scan"))
    return
end

# Solid-angle weight of the sector at pitch center `pa` [deg]; NaN for a non-finite center drops the sector.
function _sector_weight(::Type{T}, hsw, weighted) where {T}
    Δθ = T(deg2rad(hsw))
    dΩ_edge = T(Δθ * sin(Δθ))
    edge_lo, edge_hi = T(hsw), T(180 - hsw)
    return pa -> !isfinite(pa) ? T(NaN) : !weighted ? one(T) :
        ifelse(pa < edge_lo || pa > edge_hi, dΩ_edge, 2Δθ * T(sind(pa)))
end

# Writes the `(omni, para, anti, perp)` spectra of one scan `S[pitch, rest...]` into `out[rest..., t]`; `buf` holds per-sector scratch.
function _directional_scan!((omni, para, anti, perp), t, buf, S, centers, lc, weight, para_tol, perp_tol, weighted)
    T = eltype(omni)
    _finish(n, d) = weighted ? n / d : ifelse(iszero(d), T(NaN), n)
    (; dΩ) = buf
    @inbounds for p in eachindex(dΩ)
        pa = centers[p]
        dΩ[p] = weight(pa)
        buf.para[p] = T(is_parallel(pa, lc, para_tol))
        buf.anti[p] = T(is_anti_parallel(pa, lc, para_tol))
        buf.perp[p] = T(is_perpendicular(pa, lc, perp_tol))
    end
    @inbounds for E in CartesianIndices(Base.tail(axes(S)))
        no = na = np = nx = zero(T) # numerator
        do_ = da = dp = dx = zero(T) # denominator
        @simd for p in eachindex(dΩ)
            d = dΩ[p]
            sd = S[p, E] * d
            valid = isfinite(sd)
            contrib = ifelse(valid, sd, zero(sd))
            dv = ifelse(valid, d, zero(d))
            mp = buf.para[p]; ma = buf.anti[p]; mx = buf.perp[p]
            no += contrib;       do_ += dv
            np += contrib * mp;  dp += dv * mp
            na += contrib * ma;  da += dv * ma
            nx += contrib * mx;  dx += dv * mx
        end
        omni[E, t] = _finish(no, do_)
        para[E, t] = _finish(np, dp)
        anti[E, t] = _finish(na, da)
        perp[E, t] = _finish(nx, dx)
    end
    return
end

"""
    group_mean(data, groups; dims, weights = nothing)

Non-finite values are skipped; a group left without values gives `NaN`.
"""
function group_mean(data, groups; dims, weights = nothing)
    dims isa Integer && 1 <= dims <= ndims(data) || throw(ArgumentError("dims must be a data axis"))
    Base.require_one_based_indexing(data)
    n = size(data, dims)
    if !isnothing(weights)
        Base.require_one_based_indexing(weights)
        length(weights) == n || throw(DimensionMismatch("expected $n weights, got $(length(weights))"))
    end
    all(g -> checkbounds(Bool, 1:n, g), groups) || throw(ArgumentError("groups must index the $n entries along dims"))
    T = float(promote_type(eltype(data), isnothing(weights) ? Bool : eltype(weights)))
    pre, post = axes(data)[1:(dims - 1)], axes(data)[(dims + 1):end]
    out = similar(data, T, pre..., length(groups), post...)
    return _group_mean!(out, data, weights, groups, CartesianIndices(pre), CartesianIndices(post))
end

function _group_mean!(out, data, weights, groups, pre, post)
    T = eltype(out)
    den = Array{T}(undef, size(pre))
    # Each `out[pre, g, K]` is reduced right after its reset, staying cache-hot through the index loop.
    @inbounds for K in post, (g, group) in enumerate(groups)
        for I in pre
            out[I, g, K] = zero(T)
            den[I] = zero(T)
        end
        for i in group
            w = isnothing(weights) ? one(T) : T(weights[i])
            @simd for I in pre
                s = data[I, i, K]
                valid = isfinite(s)
                out[I, g, K] += ifelse(valid, T(s), zero(T)) * w
                den[I] += ifelse(valid, w, zero(T))
            end
        end
        for I in pre
            out[I, g, K] /= den[I]
        end
    end
    return out
end
