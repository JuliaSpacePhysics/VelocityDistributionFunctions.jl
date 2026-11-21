"""
    angle_distribution(data, grid, coord; B = nothing, gyro_ref = nothing, bins = 12, method = :mean) -> (; data, solid_angle, edges, centers)

All bins are `NaN` when `B` or `gyro_ref` defines no direction.
"""
function angle_distribution(data, grid, coord::Symbol; B = nothing, gyro_ref = nothing, bins = 12, method = :mean)
    method in (:mean, :sum) || throw(ArgumentError("method must be :mean or :sum, got :$method"))
    edges = _angle_edges(coord, bins)
    centers = (edges[begin:(end - 1)] .+ edges[(begin + 1):end]) ./ 2
    data, solid_angle = _bin_cells(data, grid, method, coord, edges, B, gyro_ref)
    return (; data, solid_angle, edges, centers)
end

function angle_distribution(data::_Series, grid, coord::Symbol; B = nothing, gyro_ref = nothing, kw...)
    nt = _scan_count(data, grid; B, gyro_ref)
    scans = _map_scans(i -> angle_distribution(data[i], _at(grid, i), coord; B = _at(B, i), gyro_ref = _at(gyro_ref, i), kw...), nt)
    (; edges, centers) = first(scans)
    return (; data = stack(s -> s.data, scans), solid_angle = stack(s -> s.solid_angle, scans), edges, centers)
end

function omni_spectrum(data, grid; method = :mean)
    x = angle_distribution(data, grid, :theta; bins = 1, method).data
    return x isa AbstractVector ? only(x) : dropdims(x; dims = 1)
end

function _coord_span(coord)
    coord in (:pitch, :theta) && return 180
    coord in (:gyrophase, :phi) && return 360
    throw(ArgumentError("coord must be :pitch, :gyrophase, :phi or :theta; got :$coord"))
end

_angle_edges(coord, n::Integer) = range(0.0, _coord_span(coord); length = n + 1)
function _angle_edges(coord, edges::AbstractVector)
    _coord_span(coord)
    length(edges) >= 2 && issorted(edges) ||
        throw(ArgumentError("bin edges must be ≥ 2 increasing values, got $edges"))
    return edges
end

# Bin of `x`, 0 if outside the edges
function _angle_bin(edges, x)
    n = length(edges)
    b = searchsortedlast(edges, x)
    b == n && x == last(edges) && return n - 1
    return 1 <= b < n ? b : 0
end

_wrap(x, lo) = lo + mod(x - lo, 360)

@inline function _unit(θ, φ)
    sθ, cθ = sincosd(θ)
    sφ, cφ = sincosd(φ)
    return SA[cφ * sθ, sφ * sθ, cθ]
end

# `(θ, φ) -> bin` of a cell's center direction; bin 0 everywhere if the field-aligned frame is undefined.
function _cell_binner(coord, edges, B, gyro_ref)
    coord === :theta && return (θ, φ) -> _angle_bin(edges, θ)
    coord === :phi && return (θ, φ) -> _angle_bin(edges, _wrap(φ, first(edges)))
    isnothing(B) && throw(ArgumentError(":$coord needs the magnetic field B"))
    b = SVector{3}(B)
    if coord === :pitch
        b̂ = normalize(b)
        all(isfinite, b̂) || return Returns(0)
        return (θ, φ) -> _angle_bin(edges, acosd(clamp(_unit(θ, φ) ⋅ b̂, -1, 1)))
    end
    isnothing(gyro_ref) && throw(ArgumentError(":gyrophase requires gyro_ref"))
    R = transpose(_rot_mat(b, SVector{3}(gyro_ref)))
    all(isfinite, R) || return Returns(0)
    return function (θ, φ)
        u = R * _unit(θ, φ)
        return _angle_bin(edges, _wrap(atand(u[2], u[1]), first(edges)))
    end
end

# `[bin, energy]` sums of `data` cells, weighted by solid angle and normalized for `:mean`, and the solid angle of
# the finite cells summed; `[bin]` if `grid` has no energies. `binof` is not inferable (one closure type per coord), so it is used only behind the broadcast barrier.
function _bin_cells(data, grid::EnergyAngleBins, method, coord, edges, B, gyro_ref)
    binof = _cell_binner(coord, edges, B, gyro_ref)
    nbins = length(edges) - 1
    (; energy, phi, theta, dphi, dtheta, layout) = grid
    _check_rank(data, grid)
    edim = something(findfirst(names -> :energy in names, layout), 0)
    T = promote_type(float(eltype(data)), Float64)
    num = zeros(T, nbins, edim == 0 ? 1 : size(data, edim))
    den = zero(num)
    ashape = _bshape(data, theta, dtheta, phi, dphi)
    bin = Array{Int}(undef, ashape)
    ω = Array{T}(undef, ashape)
    bin .= binof.(theta, phi)
    ω .= _solid_angle.(theta, dtheta, dphi)
    weighted = method === :mean
    _accumulate!(num, den, data, bin, ω, _cells(ashape, data), edim, weighted)
    @. num = ifelse(iszero(den), T(NaN), ifelse(weighted, num / den, num))
    return isnothing(energy) ? (vec(num), vec(den)) : (num, den)
end

# `(J, members)` for each cell `J` of an array of padded size `shape`: the data cells that broadcast onto it.
# Members span every axis (one index where the array has the axis) so `N` stays static.
function _cells(shape, data)
    return (
        (J, CartesianIndices(map((j, s, ax) -> s == 1 ? (first(ax):last(ax)) : (j:j), Tuple(J), shape, axes(data))))
            for J in CartesianIndices(shape)
    )
end

function _accumulate!(num, den, data, bin, ω, cells, edim, weighted)
    @inbounds for (J, members) in cells
        b = bin[J]
        b == 0 && continue
        w = ω[J]
        for I in members
            v = data[I]
            isfinite(v) || continue
            e = edim == 0 ? 1 : I[edim]
            num[b, e] += weighted ? v * w : v
            den[b, e] += w
        end
    end
    return
end

_solid_angle(θ, dθ, dφ) = first(_theta_factors(θ, dθ)) * deg2rad(dφ)
