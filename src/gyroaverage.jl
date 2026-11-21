# Most sub-cells along one direction of a cell: bins much finer than the cells (pitch bins near `u`) would ask for more.
const _MAX_SPLIT = 16

"""
    gyroaverage(data, grid, c1 => bins1, c2 => bins2; species, units, B, u = nothing, sc_pot = 0, oversample = 1) -> (; data, volume, edges, centers)

Return the mean phase-space density [s³/m⁶] over the measured part of each bin in the frame moving at
`u` [m/s]. Choose `:v_perp` and `:v_para` [m/s], or `:pitch` [deg] and either `:energy` [eV] or
`:speed` [m/s], in output axis order. Supply bin edges, or a bin count for pitch. `B` and `u` use the
input velocity axes. `sc_pot` [V] corrects detector energies before the frame shift.

`volume` estimates the sampled velocity-space volume [m³/s³]. Sum `data .* volume` over finite bins
for the measured density [m⁻³]. Each input cell has constant PSD. At zero frame velocity, radial
overlaps with energy or speed bins are exact. Other overlaps use sub-cell midpoints, targeting
`oversample` samples per local output bin width (up to $_MAX_SPLIT per input dimension).
Fine bins can receive zero samples even within the field of view.
"""
function gyroaverage(data, grid, (c1, bins1)::Pair{Symbol}, (c2, bins2)::Pair{Symbol}; species, units, B, u = nothing, sc_pot = 0, oversample = 1)
    names = (c1, c2)
    cylindrical = :v_perp in names && :v_para in names
    polar = :pitch in names && (:energy in names || :speed in names)
    cylindrical || polar || throw(ArgumentError("pair :v_perp with :v_para, or :pitch with :energy or :speed; got $names"))
    isfinite(oversample) && oversample > 0 || throw(ArgumentError("oversample must be finite and positive, got $oversample"))
    edges = (_gyro_edges(c1, bins1), _gyro_edges(c2, bins2))
    centers = map(e -> (e[begin:(end - 1)] .+ e[(begin + 1):end]) ./ 2, edges)
    (; mass, charge) = _species(species)
    c, p = _conversion_coeff(units, :df_m, mass)
    _check_rank(data, grid)
    num, den = _gyro_bins(data, grid, names, edges, mass, (charge / E_SI) * sc_pot, c, p, B, u, oversample)
    num .= ifelse.(iszero.(den), eltype(num)(NaN), num ./ den)
    return (; data = num, volume = den, edges, centers)
end

function gyroaverage(data::_Series, grid, c1::Pair{Symbol}, c2::Pair{Symbol}; B, u = nothing, sc_pot = 0, kw...)
    nt = _scan_count(data, grid; B, u)
    sc_pot isa Number || _check_series(sc_pot, nt, "sc_pot")
    scans = _map_scans(nt) do i
        gyroaverage(data[i], _at(grid, i), c1, c2; B = _at(B, i), u = _at(u, i), sc_pot = _at(sc_pot, i), kw...)
    end
    (; edges, centers) = first(scans)
    return (; data = stack(s -> s.data, scans), volume = stack(s -> s.volume, scans), edges, centers)
end

_gyro_edges(name, bins) = name === :pitch ? _angle_edges(name, bins) : _check_edges(bins)

# Value of coordinate `name` for plasma-frame velocity components `(w∥, w⊥, |w|²)` [m/s]; `m` [kg].
@inline function _gyro_coord(name, wpar, wperp, w2, m)
    name === :v_perp && return wperp
    name === :v_para && return wpar
    name === :speed && return sqrt(w2)
    name === :energy && return m * w2 / (2 * E_SI)
    return acosd(clamp(wpar / sqrt(w2), -1, 1))
end

# Width [m/s] of the `name` bin nearest to coordinate value `x` at plasma-frame velocity `w`.
@inline function _bin_speed_width(name, edges, x, w, m)
    b = clamp(searchsortedlast(edges, x), 1, length(edges) - 1)
    lo, hi = edges[b], edges[b + 1]
    name === :energy && return sqrt(2 * E_SI * max(hi, 0) / m) - sqrt(2 * E_SI * max(lo, 0) / m)
    name === :pitch && return norm(w) * deg2rad(hi - lo)
    return hi - lo
end

# Whether the values of coordinate `name` within distance `R` of plasma-frame velocity `w` (value `x`) can reach the edges.
@inline function _reaches(name, edges, x, w, R, m)
    s = norm(w)
    lo, hi = if name === :energy
        m * max(s - R, 0)^2 / (2 * E_SI), m * (s + R)^2 / (2 * E_SI)
    elseif name === :pitch
        s > R ? (x - asind(R / s), x + asind(R / s)) : (zero(x), oftype(x, 180))
    else
        x - R, x + R
    end
    return hi >= first(edges) && lo <= last(edges)
end

_split(x) = isnan(x) ? 1 : Int(clamp(ceil(x), 1, _MAX_SPLIT))

# `[c1, c2]` sums of `f · volume` and of `volume` over the sub-cells of the finite cells of `data`.
function _gyro_bins(data, grid::EnergyAngleBins, names, edges, m, ΔU, c, p, B, u, oversample)
    energy = _energies(grid)
    (; denergy, phi, dphi, theta, dtheta) = grid
    _bshape(data, energy, denergy, theta, dtheta, phi, dphi)
    T = promote_type(float(eltype(data)), Float64)
    num = zeros(T, length(edges[1]) - 1, length(edges[2]) - 1)
    den = zero(num)
    b̂ = normalize(SVector{3, T}(B))
    all(isfinite, b̂) || return num, den
    u0 = isnothing(u) ? zero(SVector{3, T}) : SVector{3, T}(u)
    cells = Broadcast.instantiate(Broadcast.broadcasted(tuple, data, energy, denergy, theta, dtheta, phi, dphi))
    _accumulate_gyro!(num, den, cells, CartesianIndices(data), names, edges, T(m), T(ΔU), T(c), p, b̂, u0, T(oversample))
    return num, den
end

function _accumulate_gyro!(num, den, cells, indices, names, edges, m, ΔU, c, p, b̂, u0, oversample)
    T = eltype(num)
    vscale = sqrt(2 * T(E_SI) / m)              # speed [m/s] = vscale √(E [eV])
    pitch_first = names[1] === :pitch
    pitch_axis = pitch_first ? 1 : 2
    shell_edges = if iszero(u0) && :pitch in names
        radial = pitch_first ? 2 : 1
        names[radial] === :speed ? edges[radial] : vscale .* sqrt.(max.(edges[radial], 0))
    end
    function coords(w)
        wpar, w2 = w ⋅ b̂, w ⋅ w
        wperp = sqrt(max(w2 - wpar * wpar, 0))
        return (_gyro_coord(names[1], wpar, wperp, w2, m), _gyro_coord(names[2], wpar, wperp, w2, m))
    end
    @inbounds for I in indices
        d, E, dE, θ, dθ, φ, dφ = cells[I]
        f = T(d) * c * _ipow(T(E), p)
        isfinite(f) || continue
        vlo = vscale * sqrt(max(E - dE / 2 + ΔU, 0))
        vhi = vscale * sqrt(max(E + dE / 2 + ΔU, 0))
        θlo, θhi = max(θ - dθ / 2, 0), min(θ + dθ / 2, 180)
        vhi > vlo && θhi > θlo || continue
        # Skip cells out of reach of the bins; split the others by the bins around their center.
        wc = (vlo + vhi) / 2 * SVector{3, T}(_unit(θ, φ)) - u0
        xc = coords(wc)
        sθ = θlo <= 90 <= θhi ? one(T) : max(sind(θlo), sind(θhi))
        R = (vhi - vlo) / 2 + vhi * (deg2rad(θhi - θlo) + sθ * deg2rad(dφ)) / 2   # bounds |w - wc| within the cell
        _reaches(names[1], edges[1], xc[1], wc, R, m) && _reaches(names[2], edges[2], xc[2], wc, R, m) || continue
        δ = if isnothing(shell_edges)
            min(_bin_speed_width(names[1], edges[1], xc[1], wc, m), _bin_speed_width(names[2], edges[2], xc[2], wc, m))
        else
            _bin_speed_width(:pitch, edges[pitch_axis], xc[pitch_axis], wc, m)
        end / oversample
        nv, nθ, nφ = _split((vhi - vlo) / δ), _split(vhi * deg2rad(θhi - θlo) / δ), _split(vhi * sθ * deg2rad(dφ) / δ)
        hv, hθ, hφ = (vhi - vlo) / nv, (θhi - θlo) / nθ, dφ / nφ
        if !isnothing(shell_edges)
            first_bin = max(1, searchsortedlast(shell_edges, vlo))
            last_bin = min(length(shell_edges) - 1, searchsortedfirst(shell_edges, vhi) - 1)
            first_bin > last_bin && continue
        end
        for kφ in 1:nφ
            sφ, cφ = sincosd(φ - dφ / 2 + (kφ - 0.5) * hφ)
            for kθ in 1:nθ
                θa = θlo + (kθ - 1) * hθ
                sθk, cθk = sincosd(θa + hθ / 2)
                n̂ = SVector{3, T}(sθk * cφ, sθk * sφ, cθk)
                Ω = (cosd(θa) - cosd(θa + hθ)) * deg2rad(hφ)
                if !isnothing(shell_edges)
                    pitch = _bin(edges[pitch_axis], acosd(clamp(n̂ ⋅ b̂, -1, 1)))
                    pitch == 0 && continue
                    # Pitch is constant along this ray; intersect the shell with each crossed radial bin.
                    for r in first_bin:last_bin
                        va, vb = max(vlo, shell_edges[r]), min(vhi, shell_edges[r + 1])
                        vol = (vb^3 - va^3) / 3 * Ω
                        i, j = pitch_first ? (pitch, r) : (r, pitch)
                        num[i, j] += f * vol
                        den[i, j] += vol
                    end
                    continue
                end
                for kv in 1:nv
                    va = vlo + (kv - 1) * hv
                    vb = va + hv
                    i, j = map(_bin, edges, coords((va + vb) / 2 * n̂ - u0))
                    (i == 0 || j == 0) && continue
                    vol = (vb^3 - va^3) / 3 * Ω
                    num[i, j] += f * vol
                    den[i, j] += vol
                end
            end
        end
    end
    return
end
