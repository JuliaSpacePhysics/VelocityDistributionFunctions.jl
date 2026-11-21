para_losscone(l) = min(l, 180 - l)
is_parallel(p, l, atol) = p < (para_losscone(l) - atol)
is_anti_parallel(p, l, atol) = p > (180.0 + atol - para_losscone(l))
is_perpendicular(p, l, atol) = (pl = para_losscone(l); (pl + atol) < p < (180.0 - pl - atol))


"""
    directional_spectra(S, pitch_angles, loss_cone; half_sector_width, para_tol = 0, perp_tol = 0, method = :mean)

With `l = min(loss_cone, 180° - loss_cone)`, sectors at pitch angle `θ` are
para if `θ < l - para_tol`, anti if `θ > 180° - l + para_tol`, perp if `l + perp_tol < θ < 180° - l - perp_tol`; omni takes all.
Sectors with non-finite pitch angle are skipped.

`:mean` weights sectors by solid angle `2Δθ sin θ`, or `Δθ sin Δθ` within `Δθ = half_sector_width` of 0° or 180°.
The default `Δθ` is half the median spacing of consecutive finite pitch centers, per scan.
`:sum` does not compensate for skipped sectors.
"""
function directional_spectra(S, pitch_angles, loss_cone; kw...)
    loss_cone isa Number || throw(DimensionMismatch("one scan takes a scalar loss cone"))
    pitch_angles isa AbstractVector{<:Number} || throw(DimensionMismatch("one scan takes one vector of pitch centers"))
    out = _directional_spectra((S,), pitch_angles, loss_cone; kw...)
    return map(x -> ndims(x) == 1 ? only(x) : dropdims(x; dims = ndims(x)), out)
end

function directional_spectra(S::_Series, pitch_angles, loss_cone; kw...)
    Base.require_one_based_indexing(S)
    n_t = length(S)
    pitch_angles isa AbstractVector{<:Number} || _check_series(pitch_angles, n_t, "pitch_angles")
    loss_cone isa Number || _check_series(loss_cone, n_t, "loss_cone")
    return _directional_spectra(S, pitch_angles, loss_cone; kw...)
end

# `pitch_angles` and `loss_cone` are shared or one entry per scan of `scans`; outputs are `[rest..., time]`.
function _directional_spectra(scans, pitch_angles, loss_cone; half_sector_width = nothing, para_tol = 0, perp_tol = 0, method = :mean)
    method in (:mean, :sum) || throw(ArgumentError("method must be :mean or :sum, got :$method"))
    weighted = method === :mean
    isempty(scans) && throw(ArgumentError("need at least one scan"))
    foreach(Base.require_one_based_indexing, scans)
    scan_size = size(first(scans))
    all(s -> size(s) == scan_size, scans) || throw(DimensionMismatch("scans differ in size"))
    shared = pitch_angles isa AbstractVector{<:Number}
    n_pa, spectrum_size = first(scan_size), Base.tail(scan_size)
    all(c -> axes(c, 1) == Base.OneTo(n_pa), shared ? (pitch_angles,) : pitch_angles) ||
        throw(DimensionMismatch("pitch_angles needs $n_pa centers per scan"))
    T = float(promote_type(mapreduce(eltype, promote_type, scans), shared ? eltype(pitch_angles) : mapreduce(eltype, promote_type, pitch_angles), eltype(loss_cone)))
    buf = (; dΩ = Vector{T}(undef, n_pa), para = Vector{T}(undef, n_pa), anti = Vector{T}(undef, n_pa), perp = Vector{T}(undef, n_pa))
    outs = ntuple(_ -> Array{T}(undef, spectrum_size..., length(scans)), 4)
    for (t, s) in enumerate(scans)
        centers = shared ? pitch_angles : pitch_angles[t]
        weight = _sector_weight(T, !weighted ? 0 : isnothing(half_sector_width) ? _half_spacing(centers) : half_sector_width, weighted)
        _directional_scan!(outs, t, buf, s, centers, _at(loss_cone, t), weight, para_tol, perp_tol, weighted)
    end
    return NamedTuple{(:omni, :para, :anti, :perp)}(outs)
end

function _half_spacing(centers)
    d = sort!(abs.(diff(filter(isfinite, centers))))
    isempty(d) && throw(ArgumentError("supply half_sector_width: fewer than two finite pitch centers"))
    return d[(end + 1) ÷ 2] / 2
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
"""
group_mean(data, groups; dims, weights = nothing) = _group_mean((data,), groups, weights, dims)
function group_mean(data::_Series, groups; dims, weights = nothing)
    Base.require_one_based_indexing(data)
    isempty(data) && throw(ArgumentError("need at least one scan"))
    return _group_mean(data, groups, weights, dims)
end

function _group_mean(scans, groups, weights, dims)
    d = first(scans)
    dims isa Integer && 1 <= dims <= ndims(d) || throw(ArgumentError("dims must be a data axis"))
    return _group_scans!(scans, groups, weights, dims, CartesianIndices(axes(d)[1:(dims - 1)]), CartesianIndices(axes(d)[(dims + 1):end]))
end

# Function barrier: `pre` and `post` are not inferable from a runtime `dims`. A tuple of one scan gives no time axis.
function _group_scans!(scans, groups, weights, dims, pre, post)
    d1, w1, g1 = _scan(scans, groups, weights, dims, 1)
    T = float(promote_type(eltype(d1), isnothing(w1) ? Bool : eltype(w1)))
    time = scans isa Tuple ? () : (length(scans),)
    out = similar(d1, T, axes(pre)..., length(g1), axes(post)..., time...)
    den = similar(out, axes(pre))
    for t in eachindex(scans)
        d, w, g = _scan(scans, groups, weights, dims, t, g1)
        slice = isempty(time) ? out : view(out, ntuple(_ -> :, Val(ndims(out) - 1))..., t)
        ndims(d) == ndims(slice) && all(k -> k == dims || axes(d, k) == axes(slice, k), 1:ndims(d)) ||
            throw(DimensionMismatch("scans must share the axes other than dims"))
        _group_mean!(slice, den, d, w, g, pre, post)
    end
    return out
end

# Scan `t`, its weights and index groups; the checks guard the `@inbounds` kernel.
function _scan(scans, groups, weights, dims, t, buf = nothing)
    data, weights = scans[t], _at_scan(weights, t, scans)
    Base.require_one_based_indexing(data)
    n = size(data, dims)
    if !isnothing(weights)
        Base.require_one_based_indexing(weights)
        length(weights) == n || throw(DimensionMismatch("expected $n weights, got $(length(weights))"))
    end
    return data, weights, _index_groups(groups isa Pair ? _at_scan(groups.first, t, scans) => groups.second : groups, n, buf)
end

# Weights and the values of `values => edges` are shared or one vector per scan.
_at_scan(x, t, scans) = x
_at_scan(x::AbstractVector{<:Number}, t, scans) = x
_at_scan(x::AbstractVector, t, scans) = (_check_series(x, length(scans), "values and weights"); x[t])

# Index groups of `values => edges`: entries in `[edges[g], edges[g + 1])`, the last bin closed; refills `groups`.
_index_groups(groups::Pair, n, ::Nothing) = _index_groups(groups, n, [Int[] for _ in 2:length(groups.second)])
function _index_groups((values, edges)::Pair, n, groups)
    Base.require_one_based_indexing(values)
    length(values) == n || throw(DimensionMismatch("expected $n values, got $(length(values))"))
    length(edges) >= 2 && issorted(edges) || throw(ArgumentError("bin edges must be ≥ 2 sorted values, got $edges"))
    ng = length(edges) - 1
    foreach(empty!, groups)
    for (i, v) in pairs(values)
        g = v == edges[end] ? ng : searchsortedlast(edges, v)
        1 <= g <= ng && push!(groups[g], i)
    end
    return groups
end
function _index_groups(groups, n, _)
    all(g -> checkbounds(Bool, 1:n, g), groups) || throw(ArgumentError("groups must index the $n entries along dims"))
    return groups
end

function _group_mean!(out, den, data, weights, groups, pre, post)
    T = eltype(out)
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
