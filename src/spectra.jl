para_losscone(l) = min(l, 180 - l)
is_parallel(p, l, atol) = p < (para_losscone(l) - atol)
is_anti_parallel(p, l, atol) = p > (180.0 + atol - para_losscone(l))
is_perpendicular(p, l, atol) = (pl = para_losscone(l); (pl + atol) < p < (180.0 - pl - atol))


"""
    directional_energy_spectra(spec_data, time_var, pitch_angles, loss_cone; para_tol=22.25, anti_tol=22.25)

Process 3D spectral data (pitch_angle × energy × time) to extract directional flux spectra.
This implements the same logic as pyspedas epd_l2_Espectra function.
"""
function directional_energy_spectra(S, pitch_angles, loss_cone; half_sector_width, para_tol = 0, perp_tol = 0)
    n_pa, n_E, n_t = size(S)
    T = promote_type(eltype(S), eltype(pitch_angles), eltype(loss_cone))
    omni = Matrix{T}(undef, n_E, n_t)
    para = Matrix{T}(undef, n_E, n_t)
    anti = Matrix{T}(undef, n_E, n_t)
    perp = Matrix{T}(undef, n_E, n_t)

    Δθ = T(deg2rad(half_sector_width))
    dΩ_edge = T(Δθ * sin(Δθ))
    edge_lo = T(half_sector_width)
    edge_hi = T(180 - half_sector_width)
    two_Δθ = 2Δθ
    _dΩ(pa) = ifelse(pa < edge_lo || pa > edge_hi, dΩ_edge, two_Δθ * T(sind(pa)))

    # Per-t scratch for dΩ and the three direction masks (re-used across all E).
    return @no_escape begin
        dΩ_buf = @alloc(T, n_pa)
        # pitch-angle direction masks
        mask_para = @alloc(T, n_pa)
        mask_anti = @alloc(T, n_pa)
        mask_perp = @alloc(T, n_pa)

        @inbounds for t in 1:n_t
            lc = loss_cone[t]
            for p in 1:n_pa
                pa = pitch_angles[p, t]
                dΩ_buf[p] = _dΩ(pa)
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
                    valid = !isnan(s)
                    s_eff = ifelse(valid, s, zero(s))
                    contrib = s_eff * d
                    dv = d * oftype(d, valid)
                    mp = mask_para[p]; ma = mask_anti[p]; mx = mask_perp[p]
                    no += contrib;       do_ += dv
                    np += contrib * mp;  dp += dv * mp
                    na += contrib * ma;  da += dv * ma
                    nx += contrib * mx;  dx += dv * mx
                end
                omni[E, t] = no / do_
                para[E, t] = np / dp
                anti[E, t] = na / da
                perp[E, t] = nx / dx
            end
        end
        (; omni, para, anti, perp)
    end
end

function PAspectra(S, dE, min_channels, max_channels)
    n_pa, _, n_time = size(S)
    T = promote_type(eltype(S), eltype(dE))
    # Allocate working arrays for this energy range
    numerator = similar(S, T, n_pa, n_time)
    denominator = similar(S, T, n_pa, n_time)
    return PAspectra!(S, dE, min_channels, max_channels, numerator, denominator)
end

function PAspectra!(S, dE, min_channels, max_channels, numerator, denominator)
    return map(min_channels, max_channels) do min_ch, max_ch
        _pa_kernel!(numerator, denominator, S, dE, min_ch, max_ch)
        numerator ./ denominator
    end
end

function _pa_kernel!(num, den, S, dE, e_min, e_max)
    n_pa, _, n_t = size(S)
    Tn = eltype(num)
    Td = eltype(den)
    # Init num[:, t]/den[:, t] just before reducing into them so the same cache
    # lines stay hot through the e-loop (a free-standing `fill!` is ~7% slower).
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
