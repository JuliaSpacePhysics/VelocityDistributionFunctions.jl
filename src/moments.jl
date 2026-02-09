# https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d.py

_clean_energy(E) = ifelse(E <= 0, 0.1, E)
_weight(E::T, dE::T, ΔU::T) where {T} = clamp((E + ΔU) / dE + T(0.5), zero(T), one(T))
_mask(data::T, flag) where {T} = ifelse(iszero(flag), zero(T), data)
_theta(theta, J) = theta[J]
_theta(theta::AbstractVector, J::CartesianIndex{2}) = theta[J[2]]
_phi(phi, J) = phi[J]
_phi(phi::AbstractVector, J::CartesianIndex{2}) = phi[J[1]]

include("moments/omega_weights.jl")
include("moments/helpers.jl")
include("moments/compute.jl")

"""
    plasma_moments(dists::AbstractVector, sc_pots, magfs; kw...)

Batch-compute plasma moments for multiple distributions, returning a `StructArray`.
"""
function plasma_moments(dists::AbstractVector, sc_pots; kw...)
    structT = Base.return_types(_plasma_moments, Tuple{eltype(dists), eltype(sc_pots), Nothing})[1]
    result = StructArray{structT}(undef, length(dists))
    tforeach(eachindex(dists, sc_pots)) do i
        result[i] = plasma_moments(dists[i], sc_pots[i]; kw...)
    end
    return result
end

function plasma_moments(dists::AbstractVector, sc_pots, magfs; kw...)
    structT = Base.return_types(_plasma_moments, Tuple{eltype(dists), eltype(sc_pots), eltype(magfs)})[1]
    result = StructArray{structT}(undef, length(dists))
    tforeach(eachindex(dists, sc_pots, magfs)) do i
        result[i] = plasma_moments(dists[i], sc_pots[i], magfs[i]; kw...)
    end
    return result
end

"""
    plasma_moments(dist, sc_pot=0, magf=nothing; mass=dist.mass)

Compute all plasma moments from a single particle velocity distribution.

## Arguments
- `dist`:   Distribution struct with fields `data`, `energy`, `theta`, `phi`, `mass`, `charge`.
- `sc_pot`:  Spacecraft potential `[V]`. Positive potential repels ions.
- `magf`:    Magnetic field vector for field-aligned decomposition.
            If `nothing` (default), field-aligned quantities are omitted.

## Keyword Arguments
- `mass`:    Override particle mass. Default: use `dist.mass`.
- `edim`:    Energy axis index in `data` (default `1`). When `edim ≠ 1`, a
             `permutedims` moves energy to the first axis internally.

## Returns
A `NamedTuple` of plasma moments.

## Assumptions
The input distribution must be a single-time 3D particle velocity distribution on a spherical energy-angle grid with fields:

- `data`:    Distribution values in energy flux units `[eV/(cm²·s·sr·eV)]`
- `energy` and `denergy`:  Energy bin centers and widths `[eV]` (`denergy` is optional; estimated from `energy` if absent)
- `theta` and `dtheta`:   Latitude angle bin centers and widths `[deg]` (`dtheta`/`dphi` optional; estimated from centres if absent)
- `phi` and `dphi`:     Azimuthal angle bin centers  and widths `[deg]`
- `bins`:    Bin mask (`1` = valid, `0` = masked) (optional; defaults to all-ones)
- `mass`:    Particle mass `[eV/(cm/s)²]`
- `charge`:  Particle charge `[elementary charges, signed]` (optional; defaults to `1`)

`data` and `bins` may be either 2D `(n_energy, n_angles)` (reformed/flattened)
or 3D `(n_energy, n_phi, n_theta)` (unreformed).

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
@inline plasma_moments(dist, scpot = 0, magf = nothing; kw...) =
    _plasma_moments(dist, scpot, magf; kw...)

@inline function _plasma_moments(dist, sc_pot, magf; mass = nothing, charge = nothing, edim = 1, masks = dist.bins)
    mass = @something mass dist.mass

    # Move energy axis to dim 1 when edim ≠ 1
    data = dist.data
    N = ndims(data)
    if edim != 1
        perm = ntuple(i -> i == 1 ? edim : (i <= edim ? i - 1 : i), N)
        data = permutedims(data, perm)
    end
    T = eltype(data)
    dims = size(data)
    edims = size(dist.energy)
    adims = Base.tail(dims)   # angle-only dimensions (everything after energy)
    phi, theta = dist.phi, dist.theta
    dtheta = hasproperty(dist, :dtheta) ? dist.dtheta : _bin_widths(theta)
    dphi = hasproperty(dist, :dphi) ? dist.dphi : _bin_widths(phi)
    charge = @something charge one(T)
    ΔU = T(charge * sc_pot)

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
        if isnothing(masks)
            @. fE_kernel = data * dE * _weight(energy, dE, ΔU) / energy / energy
        else
            _bins = edim != 1 ? permutedims(masks, perm) : masks
            @. fE_kernel = _mask(data, _bins) * dE * _weight(energy, dE, ΔU) / energy / energy
        end
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
