"""
    angle_distribution(data, grid, coord; B = nothing, gyro_ref = nothing, bins = 12, method = :mean, tdim = nothing) -> (; data, solid_angle, edges, centers)

For `:gyrophase`, the projection of `gyro_ref` perpendicular to `B` defines 0°. Positive angles turn toward `B × gyro_ref`.
Each cell goes whole to the bin of its center direction. Non-finite cells are skipped; bins without finite cells are `NaN`,
as are all bins when `B` or `gyro_ref` defines no direction.
"""
function angle_distribution(data, grid, coord::Symbol; B = nothing, gyro_ref = nothing, bins = 12, method = :mean, tdim = nothing)
    method in (:mean, :sum) || throw(ArgumentError("method must be :mean or :sum, got :$method"))
    edges = _angle_edges(coord, bins)
    centers = (edges[begin:(end - 1)] .+ edges[(begin + 1):end]) ./ 2
    data, solid_angle = _over_scans(data, grid, B, gyro_ref, tdim) do d, g, b, r
        _bin_cells(d, g, method, coord, edges, b, r)
    end
    return (; data, solid_angle, edges, centers)
end

function omni_spectrum(data, grid; method = :mean, tdim = nothing)
    x = angle_distribution(data, grid, :theta; bins = 1, method, tdim).data
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
function _bin_cells(data, grid::EnergyAngleGrid, method, coord, edges, B, gyro_ref)
    binof = _cell_binner(coord, edges, B, gyro_ref)
    nbins = length(edges) - 1
    (; energy, phi, theta, dphi, dtheta, layout) = grid
    ndims(data) == length(layout) ||
        throw(DimensionMismatch("data has $(ndims(data)) axes but grid layout $layout names $(length(layout))"))
    edim = something(findfirst(==(:energy), layout), 0)
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

# Tuple of arrays `f(data, grid, B, gyro_ref)` per scan along `tdim`, each stacked along a trailing axis; one call without `tdim`.
function _over_scans(f, data, grid, B, gyro_ref, tdim)
    isnothing(tdim) && return f(data, grid, B, gyro_ref)
    nt = _scan_count(data, tdim, grid; B, gyro_ref)
    scan(i) = f(selectdim(data, tdim, i), _at(grid, i), _at(B, i), _at(gyro_ref, i))
    first_scan = scan(1)
    outs = map(x -> similar(x, size(x)..., nt), first_scan)
    store!(i, xs) = foreach((o, x) -> selectdim(o, ndims(o), i) .= x, outs, xs)
    store!(1, first_scan)
    Threads.@threads :dynamic for i in 2:nt
        store!(i, scan(i))
    end
    return outs
end
