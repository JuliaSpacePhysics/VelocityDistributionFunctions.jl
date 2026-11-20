"""
    EnergyAngleBins(axes...; denergy, dphi, dtheta)

Estimated energy widths are midpoints between centers, which overestimate the end bins of a log grid (by 10% for a
20% step); prefer the instrument's `denergy`.
"""
struct EnergyAngleBins{E, P, T, DE, DP, DT, L}
    energy::E
    phi::P
    theta::T
    denergy::DE
    dphi::DP
    dtheta::DT
    layout::L
end

# Literal axis names reach inference as constants (`:phi => φ` infers with a constant `first`); the lookups recurse
# over tuples so they fold and the `EnergyAngleBins` type infers.
function EnergyAngleBins(axes::Union{Symbol, Pair}...; denergy = nothing, dphi = nothing, dtheta = nothing)
    specs = map(_axis_spec, axes)
    layout = map(first, specs)
    _check_names(layout)
    energy, denergy = _coord(specs, :energy, denergy, (nothing, nothing), nothing)
    phi, dphi = _coord(specs, :phi, dphi, (0, 360), 360)
    theta, dtheta = _coord(specs, :theta, dtheta, (90, 180), nothing)
    _check_theta(theta, dtheta)
    return EnergyAngleBins(energy, phi, theta, denergy, dphi, dtheta, layout)
end

# `(names, values)` of one axis argument; a bare name has `missing` values.
function _axis_spec(name::Symbol)
    name === :energy || throw(ArgumentError("only :energy may name an axis without values; got :$name"))
    return ((name,), (missing,))
end
function _axis_spec((names, values)::Pair)
    names isa Symbol && return ((names,), (values,))
    names isa Tuple && !isempty(names) && values isa Tuple && length(values) == length(names) ||
        throw(ArgumentError("an axis is `name => values` or `(names...) => (values...)`; got $names => $(typeof(values))"))
    return (names, values)
end

function _check_names(layout)
    names = Symbol[n for axis in layout for n in axis]
    allunique(names) && all(in((:energy, :phi, :theta)), names) ||
        throw(ArgumentError("axes name each of :energy, :phi, :theta at most once; got $layout"))
    return nothing
end

# Centers and widths of coordinate `name`: `full` if no axis names it, `missing` centers for a bare name.
# Widths are estimated only along an axis the coordinate has alone: centers on a shared axis are not ordered.
function _coord(specs, name, width, full, period)
    found = _lookup(specs, name, 1)
    if isnothing(found) || found[3] isa Missing
        isnothing(width) || throw(ArgumentError("d$name needs $name values"))
        return isnothing(found) ? full : (missing, nothing)
    end
    d, shared, values = found
    N = Val(length(specs))
    x = _place(values, d, N)
    if isnothing(width)
        !shared && size(x, d) >= 2 ||
            throw(ArgumentError("supply d$name: its centers do not vary along an axis of their own"))
        width = values isa AbstractVector ? _bin_widths(values, 1; period) : _bin_widths(x, d; period)
    end
    return (x, _place(width, d, N))
end

# `(axis, shared, values)` of coordinate `name` in `specs` from axis `d` on, or `nothing`.
_lookup(::Tuple{}, name, d) = nothing
@inline function _lookup(specs::Tuple, name, d)
    names, values = first(specs)
    i = _index(names, name, 1)
    return isnothing(i) ? _lookup(Base.tail(specs), name, d + 1) : (d, length(names) > 1, values[i])
end
_index(::Tuple{}, name, i) = nothing
@inline _index(names::Tuple, name, i) = first(names) === name ? i : _index(Base.tail(names), name, i + 1)

# A vector lies along axis `d` of `N`.
_place(x, d, ::Val) = x
_place(x::AbstractVector, d, ::Val{N}) where {N} = reshape(x, ntuple(i -> i == d ? length(x) : 1, Val(N)))

function _check_theta(theta, dtheta)
    tol = 180 * sqrt(eps(float(promote_type(eltype(theta), eltype(dtheta)))))
    inside(θ, dθ) = -tol <= θ - dθ / 2 && θ + dθ / 2 <= 180 + tol
    all(Broadcast.broadcasted(inside, theta, dtheta)) || throw(ArgumentError(
        "theta bins θ ± dθ/2 must lie within [0, 180]: θ is the polar angle of the velocity direction"
    ))
    return nothing
end
