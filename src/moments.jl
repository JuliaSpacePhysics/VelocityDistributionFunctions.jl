# https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d.py

_clean_energy(E) = ifelse(E <= 0, 0.1, E)
_weight(E::T, dE::T, ΔU::T) where {T} = clamp((E + ΔU) / dE + T(0.5), zero(T), one(T))
_mask(data::T, flag) where {T} = ifelse(iszero(flag), zero(T), data)

include("moments/omega_weights.jl")
include("moments/helpers.jl")
include("moments/compute.jl")

"""
    plasma_moments(dists::AbstractVector, sc_pots, magfs; mass=nothing)

Batch-compute plasma moments for multiple distributions, returning a `StructArray`.
"""
function plasma_moments(dists::AbstractVector, sc_pots; mass = nothing, kw...)
    structT = Base.return_types(_plasma_moments, Tuple{eltype(dists), eltype(sc_pots), Nothing})[1]
    result = StructArray{structT}(undef, length(dists))
    tforeach(eachindex(dists, sc_pots); kw...) do i
        _mass = something(mass, dists[i].mass)
        result[i] = plasma_moments(dists[i], sc_pots[i]; mass = _mass)
    end
    return result
end

function plasma_moments(dists::AbstractVector, sc_pots, magfs; mass = nothing, kw...)
    structT = Base.return_types(_plasma_moments, Tuple{eltype(dists), eltype(sc_pots), eltype(magfs)})[1]
    result = StructArray{structT}(undef, length(dists))
    tforeach(eachindex(dists, sc_pots, magfs); kw...) do i
        _mass = something(mass, dists[i].mass)
        result[i] = plasma_moments(dists[i], sc_pots[i], magfs[i]; mass = _mass)
    end
    return result
end

"""
    plasma_moments(dist, sc_pot=0, magf=nothing; mass=dist.mass)

Compute all plasma moments from a single particle velocity distribution.

## Arguments
- `dist`:   Distribution struct with fields `data`, `energy`, `theta`, `dtheta`, `phi`, `dphi`, `bins`, `mass`, `charge`.
- `sc_pot`:  Spacecraft potential `[V]`. Positive potential repels ions.
- `magf`:    Magnetic field vector for field-aligned decomposition.
            If `nothing` (default), field-aligned quantities are omitted.

## Keyword Arguments
- `mass`:    Override particle mass. Default: use `dist.mass`.

## Returns
A `NamedTuple` of plasma moments.

## Assumptions
The input distribution must be a single-time 3D particle velocity distribution on a spherical energy-angle grid with fields:

- `data`:    Distribution values in energy flux units `[eV/(cm²·s·sr·eV)]`
- `energy` and `denergy`:  Energy bin centers and widths `[eV]` (`denergy` is optional; estimated from `energy` if absent)
- `theta` and `dtheta`:   Latitude angle bin centers and widths `[deg]`
- `phi` and `dphi`:     Azimuthal angle bin centers  and widths `[deg]`
- `bins`:    Bin mask (`1` = valid, `0` = masked)
- `mass`:    Particle mass `[eV/(cm/s)²]`
- `charge`:  Particle charge `[elementary charges, signed]`

`data` and `bins` may be either 2D `(n_energy, n_angles)` (reformed/flattened)
or 3D `(n_energy, n_phi, n_theta)` (unreformed).

Energy arrays (`energy`, `denergy`) may be supplied as:
- a full array matching the shape of `data`, when energy bins vary across angles, or
- a 1D vector of length `n_energy`, when energy bins are the same for every angle.

Angle arrays (`theta`, `dtheta`, `phi`, `dphi`) may be supplied as:
- a full array matching the shape of `data` (energy axis is ignored, first slice is used), or
- an array matching the angle-only dimensions of `data` (i.e. shape `(n_angles,)` or `(n_phi, n_theta)`).

### Angular convention (SPEDAS-compatible)
- `theta`: latitude / elevation angle (−90° to 90°), **not** colatitude
- `phi`: azimuth (0° to 360°)
"""
plasma_moments(dist, scpot = 0, magf = nothing; kw...) =
    _plasma_moments(dist, scpot, magf; kw...)

function _plasma_moments(dist, sc_pot, magf; mass = nothing)
    # Angle arrays: drop energy axis if present (angles are energy-independent)
    _angle(input, n) = ndims(input) == n ? selectdim(input, 1, 1) : input

    mass = @something mass dist.mass
    T = eltype(dist.data)
    dims = size(dist.data)
    edims = size(dist.energy)
    adims = Base.tail(dims)   # angle-only dimensions (everything after energy)
    n = length(dims)
    phi, theta, dtheta, dphi = _angle.((dist.phi, dist.theta, dist.dtheta, dist.dphi), n)
    ΔU = T(dist.charge * sc_pot)

    return @no_escape begin
        energy = @alloc(T, edims...)
        e_inf = @alloc(T, edims...)
        # Full-sized work arrays
        fE_kernel = @alloc(T, dims...)
        fv_d3v_kernel = @alloc(T, dims...)
        omega = @alloc(T, 10, adims...) # Omega weights: angle-only (no energy axis)

        @. energy = _clean_energy(dist.energy)
        @. e_inf = max(energy + ΔU, zero(T))

        dE = hasproperty(dist, :denergy) ? dist.denergy :
            _compute_denergy!(@alloc(T, edims...), energy)

        omega_weights!(omega, theta, phi, dtheta, dphi)
        domega0 = reshape(selectdim(omega, 1, 1), 1, adims...)
        Omega_j = selectdim(omega, 1, 2:4)
        Omega_ij = selectdim(omega, 1, 5:10)

        # Common derived arrays
        @. fE_kernel = _mask(dist.data, dist.bins) * dE * _weight(energy, dE, ΔU) / energy / energy
        @. fv_d3v_kernel = fE_kernel * sqrt(e_inf) * domega0

        density = sqrt(mass / 2) * sum(fv_d3v_kernel)

        flux, _mftens, eflux = compute_fused_moments(fE_kernel, e_inf, Omega_j, Omega_ij)
        mftens = _mftens .* T(sqrt(2 * mass))
        velocity = flux ./ density

        qflux = compute_heat_flux(fv_d3v_kernel, theta, phi, e_inf, mass, velocity)
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
