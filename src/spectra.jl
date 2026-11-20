para_losscone(l) = min(l, 180 - l)
is_parallel(p, l, atol) = p < (para_losscone(l) - atol)
is_anti_parallel(p, l, atol) = p > (180.0 + atol - para_losscone(l))
is_perpendicular(p, l, atol) = (pl = para_losscone(l); (pl + atol) < p < (180.0 - pl - atol))


"""
    directional_energy_spectra(S, pitch_angles, loss_cone; half_sector_width = nothing, para_tol = 0, perp_tol = 0, method = :mean)

With `l = min(loss_cone, 180° - loss_cone)`, sectors at pitch angle `θ` are
para if `θ < l - para_tol`, anti if `θ > 180° - l + para_tol`, perp if `l + perp_tol < θ < 180° - l - perp_tol`; omni takes all.
Sectors with non-finite flux or pitch angle are skipped; a direction left without sectors gives `NaN`.

`:mean` weights sectors by solid angle `2Δθ sin θ`, or `Δθ sin Δθ` within `Δθ = half_sector_width` of 0° or 180°.
`:sum` does not compensate for skipped sectors.
"""
function directional_energy_spectra(S, pitch_angles, loss_cone; half_sector_width = nothing, para_tol = 0, perp_tol = 0, method = :mean)
    method in (:mean, :sum) || throw(ArgumentError("method must be :mean or :sum, got :$method"))
    weighted = method === :mean
    weighted && isnothing(half_sector_width) && throw(ArgumentError("method = :mean requires half_sector_width"))
    hsw = weighted ? half_sector_width : 0
    n_pa, n_E, n_t = size(S)
    T = float(promote_type(eltype(S), eltype(pitch_angles), eltype(loss_cone)))
    omni = Matrix{T}(undef, n_E, n_t)
    para = Matrix{T}(undef, n_E, n_t)
    anti = Matrix{T}(undef, n_E, n_t)
    perp = Matrix{T}(undef, n_E, n_t)

    Δθ = T(deg2rad(hsw))
    dΩ_edge = T(Δθ * sin(Δθ))
    edge_lo = T(hsw)
    edge_hi = T(180 - hsw)
    two_Δθ = 2Δθ
    _dΩ(pa) = ifelse(pa < edge_lo || pa > edge_hi, dΩ_edge, two_Δθ * T(sind(pa)))
    _finish(n, d) = weighted ? n / d : ifelse(iszero(d), T(NaN), n)

    dΩ_buf = Vector{T}(undef, n_pa)
    mask_para = Vector{T}(undef, n_pa)
    mask_anti = Vector{T}(undef, n_pa)
    mask_perp = Vector{T}(undef, n_pa)

    @inbounds for t in 1:n_t
        lc = loss_cone[t]
        for p in 1:n_pa
            pa = pitch_angles[p, t]
            dΩ_buf[p] = !isfinite(pa) ? T(NaN) : weighted ? _dΩ(pa) : one(T) # NaN weight drops the sector
            mask_para[p] = T(is_parallel(pa, lc, para_tol))
            mask_anti[p] = T(is_anti_parallel(pa, lc, para_tol))
            mask_perp[p] = T(is_perpendicular(pa, lc, perp_tol))
        end
        for E in 1:n_E
            no = na = np = nx = zero(T) # numerator
            do_ = da = dp = dx = zero(T) # denominator
            @simd for p in 1:n_pa
                s = S[p, E, t]
                d = dΩ_buf[p]
                sd = s * d
                valid = isfinite(sd)
                contrib = ifelse(valid, sd, zero(sd))
                dv = ifelse(valid, d, zero(d))
                mp = mask_para[p]; ma = mask_anti[p]; mx = mask_perp[p]
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
    end
    return (; omni, para, anti, perp)
end

function pitch_angle_spectra(S, dE, min_channels, max_channels)
    n_pa, _, n_time = size(S)
    T = promote_type(eltype(S), eltype(dE))
    numerator = similar(S, T, n_pa, n_time)
    denominator = similar(S, T, n_pa, n_time)
    return map(min_channels, max_channels) do min_ch, max_ch
        _pa_kernel!(numerator, denominator, S, dE, min_ch, max_ch)
        numerator ./ denominator
    end
end

function _pa_kernel!(num, den, S, dE, e_min, e_max)
    n_pa, _, n_t = size(S)
    Tn = eltype(num)
    Td = eltype(den)
    # Init num[:, t]/den[:, t] just before reducing into them to keep them cache-hot through the e-loop.
    @inbounds for t in 1:n_t
        @simd for p in 1:n_pa
            num[p, t] = zero(Tn)
            den[p, t] = zero(Td)
        end
        for e in e_min:e_max
            de = dE[e]
            @simd for p in 1:n_pa
                s = S[p, e, t]
                valid = isfinite(s)
                num[p, t] += ifelse(valid, s, zero(s)) * de
                den[p, t] += de * oftype(de, valid)
            end
        end
    end
    return num, den
end
