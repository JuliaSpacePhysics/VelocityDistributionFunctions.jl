# https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d.py

_clean_energy(E) = ifelse(E <= 0, 0.1, E)
_weight(E::T, dE::T, ΔU::T) where {T} = clamp((E + ΔU) / dE + T(0.5), zero(T), one(T))

include("moments/omega_weights.jl")
include("moments/helpers.jl")
include("moments/compute.jl")

# ── High-level time-series interface ─────────────────────────────────────────

"""
    tmoments(data, theta, phi, energy, sc_pot, magf; tdim, edim, kw...)

Compute plasma moments for every timestep in a multi-time raw distribution array.

Loops over the time dimension `tdim`, calling [`plasma_moments`](@ref) for each slice, and returns a `StructArray`.

## Keyword Arguments
- `tdim`:        Which axis of `data` is time (**required**).

## Returns
A `StructArray` of moment `NamedTuple`s, one per timestep.

## Example
```julia
tmoments(data, theta, phi, energy; species=:H, tdim=1, edim=4)
```
"""
@inline function tmoments(data, theta, phi, energy, sc_pot = 0, magf = nothing; tdim = ndims(data), edim = 1, kw...)
    nt = size(data, tdim)
    # After removing tdim, edim shifts down by 1 if it was after tdim
    edim_slice = edim > tdim ? edim - 1 : edim

    result = tmap(1:nt) do i
        efluxi = selectdim(data, tdim, i)
        disti = (; data = efluxi, theta = _vector(theta, i), phi = _vector(phi, i), energy = _vector(energy, i))
        plasma_moments(
            disti, _scalar(sc_pot, i), _vector(magf, i);
            edim = edim_slice, kw...
        )
    end
    return StructArray(result)
end

# ── High-level single-timestep interface ─────────────────────────────────────
"""
    plasma_moments(dist, sc_pot=0, magf=nothing; mass, charge, edim)

Compute all plasma moments from a single distribution on a spherical energy-angle grid.

Handles unit conversion, species mass/charge lookup, and coordinate transforms internally.

## Arguments
- `dist`:   Distribution struct with fields `data`, `energy`, `theta`, `phi`, `mass`, `charge`.
- `sc_pot`:  Spacecraft potential `[V]`. Positive potential repels ions.
- `magf`:    Magnetic field vector for field-aligned decomposition.
            If `nothing` (default), field-aligned quantities are omitted.

## Keyword Arguments
- `species`: Species symbol (`:H`, `:He`, `:O`, `:e`). Determines mass, charge, and the `A` parameter for unit conversion. Defaults to `:H`.
- `edim`: Energy axis index in `data` (default `1`). When `edim ≠ 1`, a `permutedims` moves energy to the first axis internally.
- `units`: Input unit system (`:eflux`, `:flux`, `:df_km`, `:df_cm`). Default `:eflux` (no conversion).

## Returns
A `NamedTuple` of plasma moments.

## Assumptions
The input distribution must have the following fields:

- `data`:    Distribution values in energy flux units `[eV/(cm²·s·sr·eV)]`
- `energy` and `denergy`:  Energy bin centers and widths `[eV]` (`denergy` is optional; estimated from `energy` if absent)
- `theta` and `dtheta`:   Latitude angle bin centers and widths `[deg]` (`dtheta`/`dphi` optional; estimated from centres if absent)
- `phi` and `dphi`:     Azimuthal angle bin centers  and widths `[deg]`
- `mass`:    Particle mass `[eV/(cm/s)²]`
- `charge`:  Particle charge `[elementary charges, signed]` (optional; defaults to `1`)

`data` may be either 2D `(n_energy, n_angles)` (reformed/flattened) or 3D `(n_energy, n_phi, n_theta)` (unreformed).

Energy arrays (`energy`, `denergy`) may be supplied as:
- a full array matching the shape of `data`, when energy bins vary across angles, or
- a 1D vector of length `n_energy`, when energy bins are the same for every angle.

Angle arrays (`theta`, `dtheta`, `phi`, `dphi`) may be supplied as:
- an array matching the angle-only dimensions of `data` (i.e. shape `(n_angles,)` or `(n_phi, n_theta)`).
- a 1D vector of length `n_theta` for `theta`, and `n_phi` for `phi`, when `data` is 3D.

### Angular convention (SPEDAS-compatible)
- `theta`: latitude / elevation angle (−90° to 90°), **not** colatitude
- `phi`: azimuth (0° to 360°)
"""
@inline function plasma_moments(dist, scpot = 0, magf = nothing; species = :H, units = :eflux, edim = 1, kw...)
    s = species_info(species)
    dtheta = get(dist, :dtheta, nothing)
    dphi = get(dist, :dphi, nothing)
    dE = get(dist, :denergy, nothing)
    c, E_exp = _conversion_coeff(units, :eflux, s.A)
    return _plasma_moments(dist.data, dist.theta, dist.phi, dist.energy, scpot, magf; coefficient = c, E_exp, mass = s.mass, charge = s.charge, dtheta, dphi, dE, edim, kw...)
end

_osize(data, dim) = ntuple(i -> size(data, i < dim ? i : i + 1), ndims(data) - 1)

@inline function _plasma_moments(data, theta, phi, _energy, sc_pot, magf; coefficient = 1.0, E_exp = 0, mass = _PROTON_MASS, charge = 1.0, dtheta = nothing, dphi = nothing, dE = nothing, edim = 1)
    T = eltype(data)
    edims = size(_energy)
    adims = _osize(data, edim)   # angle-only dimensions (everything after energy)
    thetadims = size(theta)
    phidims = size(phi)
    ΔU = T(charge * sc_pot)

    return @no_escape begin
        Escale = @alloc(T, edims...)
        energy = @alloc(T, edims...)
        e_inf = @alloc(T, edims...)
        v_inf = @alloc(T, edims...)
        dtheta = @something(dtheta, _bin_widths!(@alloc(T, thetadims...), theta))
        dphi = @something(dphi, _bin_widths!(@alloc(T, phidims...), phi, 360))
        Omega = @alloc(T, 10, adims...) # Omega weights: angle-only (no energy axis)

        @. energy = _clean_energy(_energy)
        dE = @something(dE, _compute_denergy!(@alloc(T, edims...), energy))

        omega_weights!(Omega, theta, phi, dtheta, dphi)

        # Common derived arrays
        @. Escale = coefficient * dE * _weight(energy, dE, ΔU) * energy^(E_exp - 2)
        @. e_inf = max(energy + ΔU, zero(T))
        _den, flux, _mftens, eflux = compute_fused_moments(data, Escale, e_inf, Omega; edim)
        density = sqrt(mass / 2) * _den
        mftens = _mftens .* T(sqrt(2 * mass))
        velocity = flux ./ density

        @. v_inf = sqrt(2 / mass * e_inf) # v_inf
        qflux = compute_heat_flux(data, Omega, theta, phi, v_inf, Escale, mass, velocity; edim)
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
