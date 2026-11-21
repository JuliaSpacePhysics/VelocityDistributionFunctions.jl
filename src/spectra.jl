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
function group_mean(data, groups; dims, weights = nothing)
    groups = _index_groups(groups)
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

function group_mean(data::_Series, groups; dims, weights = nothing)
    Base.require_one_based_indexing(data)
    nt = length(data)
    nt > 0 || throw(ArgumentError("need at least one scan"))
    per_scan(x) = x isa AbstractVector && !(x isa AbstractVector{<:Number}) && (_check_series(x, nt, "values and weights"); true)
    values_vary = groups isa Pair && per_scan(groups.first)
    weights_vary = !isnothing(weights) && per_scan(weights)
    scan_groups(i) = values_vary ? groups.first[i] => groups.second : groups
    return stack(i -> group_mean(data[i], scan_groups(i); dims, weights = weights_vary ? weights[i] : weights), 1:nt)
end

# Index groups of `values => edges`: entries in `[edges[g], edges[g + 1])`, the last bin closed.
_index_groups(groups) = groups
function _index_groups((values, edges)::Pair)
    length(edges) >= 2 && issorted(edges) || throw(ArgumentError("bin edges must be ≥ 2 increasing values, got $edges"))
    return [findall(v -> edges[g] <= v < edges[g + 1] || (g == length(edges) - 1 && v == edges[end]), values) for g in 1:(length(edges) - 1)]
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
