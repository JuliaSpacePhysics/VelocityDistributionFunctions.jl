# https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d.py

_clean_energy(E) = ifelse(E <= 0, 0.1, E)
_weight(E::T, dE::T, ΔU::T) where {T} = clamp((E + ΔU) / dE + T(0.5), zero(T), one(T))

include("moments/omega_weights.jl")
include("moments/helpers.jl")
include("moments/compute.jl")

"""
    plasma_moments(data, grid; species, units, sc_pot = 0, B = nothing, tdim = nothing)

Moments of one scan as a NamedTuple. With `tdim`, return a StructArray with one record per scan along that axis.
Use one grid and spacecraft potential for all scans or one of each per scan. Supply series fields as one `B` vector per scan.
Align ancillary measurements to the scan intervals before calling.

- `data` is in `units`: `:eflux` [eV/(cm²·s·sr·eV)], `:flux` [1/(cm²·s·sr·eV)], `:df_cm` [s³/cm⁶], `:df_km` [s³/km⁶] or `:df_m` [s³/m⁶].
- `species` is `:H`, `:He` (α), `:O` (¹⁶O⁺), `:e`, or a `PlasmaPopulations.Particle`.
- `sc_pot` [V] shifts energies by `charge * sc_pot`.
- Non-finite `data` bins contribute zero; a scan without finite bins gives NaN moments.
- Output NamedTuple is CGS with energies in eV: `density` [cm⁻³], `velocity`, `vthermal` [cm/s], `flux` [cm⁻² s⁻¹],
  `eflux`, `qflux` [eV cm⁻² s⁻¹], `mftens`, `ptens` [eV cm⁻³] as `(xx, yy, zz, xy, xz, yz)`, `ttens`, `avgtemp`, `t3` [eV].
- With field `B`, adds `magt3` [eV] `(⊥, ⊥, ∥)` and the symmetry axis `symm` with `symm_theta`, `symm_phi`, `symm_ang` [deg].
"""
function plasma_moments(data, grid; species, units, sc_pot = 0, B = nothing, tdim = nothing)
    (; mass, charge) = _species(species)
    coefficient, E_exp = _conversion_coeff(units, :eflux, mass)
    function scan(d, g::EnergyAngleBins, pot, field)
        ndims(d) == length(g.layout) ||
            throw(DimensionMismatch("scan has $(ndims(d)) axes but grid axes $(g.layout) name $(length(g.layout))"))
        return _plasma_moments(d, g, pot, field; coefficient, E_exp, mass, charge)
    end
    isnothing(tdim) && return scan(data, grid, sc_pot, B)
    nt = _scan_count(data, tdim, grid; B)
    sc_pot isa Number || _check_series(sc_pot, nt, "sc_pot")
    moment_at(i) = scan(selectdim(data, tdim, i), _at(grid, i), _at(sc_pot, i), _at(B, i))
    first_result = moment_at(1)
    result = StructArray{typeof(first_result)}(undef, nt)
    result[1] = first_result
    Threads.@threads :dynamic for i in 2:nt
        result[i] = moment_at(i)
    end
    return result
end

function _scan_count(data, tdim, grid; vectors...)
    nt = _scan_count(data, tdim)
    nt > 0 || throw(ArgumentError("need at least one scan"))
    grid isa EnergyAngleBins || _check_series(grid, nt, "grid")
    for (name, v) in pairs(vectors)
        isnothing(v) && continue
        _check_series(v, nt, name)
        all(x -> (x isa AbstractVector || x isa Tuple) && length(x) == 3, v) ||
            throw(DimensionMismatch("$name needs one three-component vector per scan"))
    end
    return nt
end

function _scan_count(data, tdim)
    tdim isa Integer && 1 <= tdim <= ndims(data) || throw(ArgumentError("tdim must be a data axis"))
    return size(data, tdim)
end

function _check_series(x, nt, name)
    x isa AbstractVector && axes(x, 1) == Base.OneTo(nt) ||
        throw(DimensionMismatch("$name needs one entry for each of $nt scans"))
end

_at(x, i) = x
_at(x::AbstractVector, i) = x[i]

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
    energy isa Union{Nothing, Missing} && throw(ArgumentError("moments need grid energies"))
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
