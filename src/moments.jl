# https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d.py

_clean_energy(E) = ifelse(E <= 0, 0.1, E)
_weight(E::T, dE::T, ΔU::T) where {T} = clamp((E + ΔU) / dE + T(0.5), zero(T), one(T))

include("moments/omega_weights.jl")
include("moments/helpers.jl")
include("moments/compute.jl")

"""
    EnergyAngleGrid(energy, phi, theta; layout, denergy, dphi, dtheta)

Bin centers and widths of a spherical velocity-space grid: `energy` [eV] and the velocity direction `phi`, `theta` [deg].

`layout` names the data axes, from `:energy`, `:phi`, `:theta` and `:angle` (flattened direction), e.g. `(:phi, :theta, :energy)`.
A vector lies along its named axis; other arrays broadcast against the data. Missing widths are estimated along each
coordinate's own axis, so a flattened `:angle` axis needs them supplied. Estimated energy widths are midpoints between
centers, which overestimate the end bins of a log grid (by 10% for a 20% step); prefer the instrument's `denergy`.
"""
struct EnergyAngleGrid{E, P, T, DE, DP, DT, N}
    energy::E
    phi::P
    theta::T
    denergy::DE
    dphi::DP
    dtheta::DT
    layout::NTuple{N, Symbol}
end

function EnergyAngleGrid(energy, phi, theta; layout, denergy = nothing, dphi = nothing, dtheta = nothing)
    allunique(layout) && all(in((:energy, :phi, :theta, :angle)), layout) ||
        throw(ArgumentError("layout names distinct axes from :energy, :phi, :theta, :angle; got $layout"))
    layout = Tuple(layout)
    place_energy(x) = _place(x, (:energy,), layout)
    place_phi(x) = _place(x, (:phi, :angle), layout)
    place_theta(x) = _place(x, (:theta, :angle), layout)
    phi, theta = place_phi(phi), place_theta(theta)
    if isnothing(energy)
        :energy in layout && throw(ArgumentError("layout $layout has an :energy axis, so energy is required"))
    else
        energy = place_energy(energy)
        denergy = @something(place_energy(denergy), _grid_widths(energy, :energy, layout))
    end
    dphi = @something(place_phi(dphi), _grid_widths(phi, :phi, layout; period = 360))
    dtheta = @something(place_theta(dtheta), _grid_widths(theta, :theta, layout))
    _check_theta(theta, dtheta)
    return EnergyAngleGrid(energy, phi, theta, denergy, dphi, dtheta, layout)
end

"""
    plasma_moments(data, grid; species, units, sc_pot = 0, B = nothing)

Moments of one distribution `data` on `grid`.

- `data` is in `units`: `:eflux` [eV/(cm²·s·sr·eV)], `:flux` [1/(cm²·s·sr·eV)], `:df_cm` [s³/cm⁶], `:df_km` [s³/km⁶] or `:df_m` [s³/m⁶].
- `species` is `:H`, `:He` (α), `:O` (¹⁶O⁺), `:e`, or a `PlasmaPopulations.Particle`.
- `sc_pot` [V] shifts energies by `charge * sc_pot`.
- Non-finite `data` bins contribute zero; a scan without finite bins gives NaN moments.
- Output NamedTuple is CGS with energies in eV: `density` [cm⁻³], `velocity`, `vthermal` [cm/s], `flux` [cm⁻² s⁻¹],
  `eflux`, `qflux` [eV cm⁻² s⁻¹], `mftens`, `ptens` [eV cm⁻³] as `(xx, yy, zz, xy, xz, yz)`, `ttens`, `avgtemp`, `t3` [eV].
- With field `B`, adds `magt3` [eV] `(⊥, ⊥, ∥)` and the symmetry axis `symm` with `symm_theta`, `symm_phi`, `symm_ang` [deg].
"""
function plasma_moments(data, grid::EnergyAngleGrid; species, units, sc_pot = 0, B = nothing)
    ndims(data) == length(grid.layout) ||
        throw(DimensionMismatch("data has $(ndims(data)) axes but grid layout $(grid.layout) names $(length(grid.layout))"))
    (; mass, charge) = _species(species)
    coefficient, E_exp = _conversion_coeff(units, :eflux, mass)
    return _plasma_moments(data, grid, sc_pot, B; coefficient, E_exp, mass, charge)
end

"""
    tmoments(data, grid; species, units, sc_pot = 0, B = nothing, tdim = ndims(data))::StructArray

Plasma moments of each slice of `data` along its time axis `tdim`; grid arrays broadcast against a slice. Align ancillary measurements to the scan intervals before calling.
`grid`, `sc_pot` and `B` are a vector with one value per time, or, for `grid` and `sc_pot`, one value for all times.
"""
function tmoments(data, grid; species, units, sc_pot = 0, B = nothing, tdim = ndims(data))
    nt = _scan_count(data, tdim, grid; B)
    sc_pot isa Number || _check_series(sc_pot, nt, "sc_pot")
    moment_at(i) = plasma_moments(
        selectdim(data, tdim, i), _at(grid, i);
        sc_pot = _at(sc_pot, i), B = _at(B, i), species, units
    )
    first_result = moment_at(1)
    result = StructArray{typeof(first_result)}(undef, nt)
    result[1] = first_result
    Threads.@threads :dynamic for i in 2:nt
        result[i] = moment_at(i)
    end
    return result
end


function _scan_count(data, tdim, grid; vectors...)
    1 <= tdim <= ndims(data) || throw(ArgumentError("tdim must be a data axis"))
    nt = size(data, tdim)
    nt > 0 || throw(ArgumentError("need at least one scan"))
    grid isa EnergyAngleGrid || _check_series(grid, nt, "grid")
    for (name, v) in pairs(vectors)
        isnothing(v) && continue
        _check_series(v, nt, name)
        all(x -> (x isa AbstractVector || x isa Tuple) && length(x) == 3, v) ||
            throw(DimensionMismatch("$name needs one three-component vector per scan"))
    end
    return nt
end

function _check_series(x, nt, name)
    x isa AbstractVector && axes(x, 1) == Base.OneTo(nt) ||
        throw(DimensionMismatch("$name needs one entry for each of $nt scans"))
end

_at(x, i) = x
_at(x::AbstractVector, i) = x[i]

# A vector lies along the first `layout` axis named in `names`.
_place(x, names, layout) = x
function _place(x::AbstractVector, names, layout)
    d = findfirst(in(names), layout)
    isnothing(d) && throw(ArgumentError("layout $layout has no $(join(names, " or ")) axis"))
    return reshape(x, ntuple(i -> i == d ? length(x) : 1, length(layout)))
end

function _check_theta(theta, dtheta)
    tol = 180 * sqrt(eps(float(promote_type(eltype(theta), eltype(dtheta)))))
    inside(θ, dθ) = -tol <= θ - dθ / 2 && θ + dθ / 2 <= 180 + tol
    all(Broadcast.broadcasted(inside, theta, dtheta)) || throw(ArgumentError(
        "theta bins θ ± dθ/2 must lie within [0, 180]: θ is the polar angle of the velocity direction"
    ))
    return nothing
end

# Only along the coordinate's own axis: centers of a flattened `:angle` axis are not ordered.
function _grid_widths(x, name, layout; period = nothing)
    d = findfirst(==(name), layout)
    (isnothing(d) || size(x, d) < 2) &&
        throw(ArgumentError("supply the bin widths of $name: its centers do not vary along a :$name axis of layout $layout"))
    return _bin_widths(x, d; period)
end

# Size of the broadcast of `xs` padded to `ndims(data)`; each must broadcast against `data`.
function _bshape(data, xs...)
    ax = Broadcast.broadcast_shape(map(_ -> Base.OneTo(1), axes(data)), map(axes, xs)...)
    Broadcast.broadcast_shape(axes(data), ax) == axes(data) ||
        throw(DimensionMismatch("grid arrays of sizes $(map(size, xs)) do not broadcast against data of size $(size(data))"))
    return map(length, ax)
end

# Energy factors scale·E∞^p, p = 1/2, 1, 3/2, 2, of the density, flux, mftens and eflux sums.
@inline function _energy_weights(E, dE, ΔU, coefficient, E_exp)
    E = _clean_energy(E)
    scale = coefficient * dE * _weight(E, dE, ΔU) * E^(E_exp - 2)
    e_inf = max(E + ΔU, zero(E))
    sqrte = sqrt(e_inf)
    w = scale * e_inf
    return (scale * sqrte, w, w * sqrte, w * e_inf)
end

# `Array` over a bump-allocated buffer: `UnsafeArray` stores carry no aliasing information, so the scatter loop
# of `_energy_sums!` would not vectorize.
_as_array(A) = unsafe_wrap(Array, pointer(A), size(A))

function _plasma_moments(data, grid, sc_pot, magf; coefficient, E_exp, mass, charge)
    (; energy, denergy, phi, dphi, theta, dtheta) = grid
    isnothing(energy) && throw(ArgumentError("moments need grid energies"))
    # At least Float64: central moments are differences of lab-frame moments, amplifying rounding by
    # |mftens|/|ptens| and |eflux|/|qflux| (~10² and ~10⁵ in the solar wind).
    T = promote_type(eltype(data), Float64)
    eshape = _bshape(data, energy, denergy)
    θshape, φshape = _bshape(data, theta, dtheta), _bshape(data, phi, dphi)
    ashape = map(max, θshape, φshape)
    ΔU = T(charge * sc_pot)
    A = promote_type(eltype(theta), eltype(dtheta), eltype(phi), eltype(dphi), Float64)

    return @no_escape begin
        θf = @alloc(NTuple{6, A}, θshape...)
        φf = @alloc(NTuple{6, A}, φshape...)
        omega = @alloc(NTuple{10, A}, ashape...)
        @. θf = _theta_factors(A(theta), A(dtheta))
        @. φf = _phi_factors(A(phi), A(dphi))
        @. omega = _omega(θf, φf)
        W = _as_array(@alloc(NTuple{4, T}, eshape...))
        @. W = _energy_weights(T(energy), T(denergy), ΔU, coefficient, E_exp)
        S = _as_array(@alloc(T, length(omega), 4))
        _den, flux, _mftens, eflux = compute_fused_moments(data, W, omega, S)
        density = sqrt(mass / 2) * _den
        mftens = _mftens .* T(sqrt(2 * mass))
        velocity = flux ./ density

        qflux = compute_heat_flux(eflux, mftens, velocity, density, mass)
        ptens = compute_pressure_tensor(mftens, velocity, flux, mass)
        temp = compute_temperature(ptens, density, mass)

        core = (;
            density, flux, velocity, eflux, qflux, mftens, ptens,
            temp.ttens, temp.avgtemp, temp.vthermal, temp.t3,
        )

        isnothing(magf) ? core : begin
                fa = compute_field_aligned(temp.ttens, temp.t3evec, velocity, magf)
                (; core..., fa...)
            end
    end
end
