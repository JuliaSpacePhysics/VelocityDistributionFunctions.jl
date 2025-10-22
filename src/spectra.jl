# It will mutate solid_angle
function sum_valid!(S, dΩ_valid, solid_angle)
    @sum flux[E, t] := _finite(S[p, E, t]) * dΩ_valid[p, t]
    @sum solid_angle[E, t] = dΩ_valid[p, t] * !isnan(S[p, E, t])
    return flux ./= solid_angle
end

# Fast when nan are replaced with 0
function sum_valid_fast!(S, dΩ_valid, solid_angle)
    @sum flux[E, t] := S[p, E, t] * dΩ_valid[p, t]
    @sum solid_angle[E, t] = dΩ_valid[p, t] * iszero(S[p, E, t])
    return flux ./= solid_angle
end

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
    n_pa, n_energy, n_time = size(S)
    T = promote_type(eltype(S), eltype(pitch_angles), eltype(loss_cone))
    return @no_escape begin
        Ω = @alloc(T, n_energy, n_time)
        dΩ = @alloc(T, n_pa, n_time)
        dΩ_valid = @alloc(T, n_pa, n_time)

        # Calculate differential solid angle elements for each pitch angle bin
        # Special handling for edge sectors
        in_edge(pa) = pa < half_sector_width || pa > (180 - half_sector_width)
        Δθ = deg2rad(half_sector_width)
        dΩ_edge = Δθ * sin(Δθ)
        @sum dΩ[pC, t] = in_edge(pitch_angles[pC, t]) ? dΩ_edge : 2Δθ * sind(pitch_angles[pC, t])

        # 1. Omnidirectional flux
        omni = sum_valid!(S, dΩ, Ω)
        # 2. Parallel flux
        @sum dΩ_valid[pC, t] = dΩ[pC, t] * is_parallel(pitch_angles[pC, t], loss_cone[t], $para_tol)
        para = sum_valid!(S, dΩ_valid, Ω)
        # 3. Antiparallel flux
        @sum dΩ_valid[pC, t] = dΩ[pC, t] * is_anti_parallel(pitch_angles[pC, t], loss_cone[t], $para_tol)
        anti = sum_valid!(S, dΩ_valid, Ω)
        # 4. Perpendicular flux
        @sum dΩ_valid[pC, t] = dΩ[pC, t] * is_perpendicular(pitch_angles[pC, t], loss_cone[t], $perp_tol)
        perp = sum_valid!(S, dΩ_valid, Ω)

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
        dE_valid = @view dE[min_ch:max_ch]
        S_valid = @view S[:, min_ch:max_ch, :]
        @sum numerator[p, t] = _finite(S_valid[p, e, t]) * dE_valid[e]
        @sum denominator[p, t] = dE_valid[e] * isfinite(S_valid[p, e, t])
        numerator ./ denominator
    end
end
