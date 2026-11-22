# https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d.py

_clean_energy(E) = ifelse(E <= 0, 0.1, E)
_weight(E::T, dE::T, ΔU::T) where {T} = clamp((E + ΔU) / dE + T(0.5), zero(T), one(T))

include("moments/omega_weights.jl")
include("moments/helpers.jl")
include("moments/compute.jl")

"""
    plasma_moments(data, grid; species, units, sc_pot = 0, B = nothing)

Align ancillary measurements to the scan intervals before calling.
"""
function plasma_moments(data, grid; species, units, sc_pot = 0, B = nothing)
    (; mass, charge) = _species(species)
    c, E_exp = _conversion_coeff(units, :eflux, mass)
    coefficient = 1.0e4 * c / E_SI^E_exp  # eflux [1/(m² s sr)] for energies in J
    _check_rank(data, grid)
    return _plasma_moments(data, grid, sc_pot, B; coefficient, E_exp, mass, charge)
end

function plasma_moments(data::_Series, grid; sc_pot = 0, B = nothing, kw...)
    nt = _scan_count(data, grid; B)
    sc_pot isa Number || _check_series(sc_pot, nt, "sc_pot")
    return StructArray(_map_scans(i -> plasma_moments(data[i], _at(grid, i); sc_pot = _at(sc_pot, i), B = _at(B, i), kw...), nt))
end

# Size of the broadcast of `xs` padded to `ndims(data)`; each must broadcast against `data`.
function _bshape(data, xs...)
    ax = Broadcast.broadcast_shape(map(_ -> Base.OneTo(1), axes(data)), map(axes, xs)...)
    Broadcast.broadcast_shape(axes(data), ax) == axes(data) ||
        throw(DimensionMismatch("grid arrays of sizes $(map(size, xs)) do not broadcast against data of size $(size(data))"))
    return map(length, ax)
end

# Energy factors scale·E∞^p, p = 1/2, 1, 3/2, 2, of the density, flux, mftens and eflux sums; `E`, `dE` [eV], `ΔU` [J].
@inline function _energy_weights(E, dE, ΔU, coefficient, E_exp)
    E = _clean_energy(E) * oftype(E, E_SI)
    dE *= oftype(dE, E_SI)
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
    energy = _energies(grid)
    (; denergy, phi, dphi, theta, dtheta) = grid
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
        density = T(sqrt(mass / 2)) * _den
        mftens = _symmetric(_mftens) .* T(sqrt(2 * mass))
        velocity = flux ./ density

        qflux = compute_heat_flux(eflux, mftens, velocity, density, mass)
        ptens = compute_pressure_tensor(mftens, flux, density, mass)
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
