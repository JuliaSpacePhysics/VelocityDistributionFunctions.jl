# https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d.py

_clean_energy(E) = ifelse(E <= 0, 0.1, E)
_weight(E::T, dE::T, ΔU::T) where {T} = clamp((E + ΔU) / dE + T(0.5), zero(T), one(T))

include("moments/omega_weights.jl")
include("moments/helpers.jl")
include("moments/compute.jl")

"""
    EnergyAngleGrid(energy, phi, theta; denergy = nothing, dphi = nothing, dtheta = nothing)

Bin centers of a spherical velocity-space grid: `energy` [eV] and the velocity direction `phi`, `theta` [deg].
Missing bin widths are estimated from the centers.

For data with energy axis `edim` and angle axes either one flattened axis or `(φ, θ)`:
- `energy`, `denergy` are vectors along energy or arrays shaped like the data.
- Angle arrays are shaped like the angle axes, or are vectors along `φ` and `θ` when those are separate axes.
"""
struct EnergyAngleGrid{E, P, T, DE, DP, DT}
    energy::E
    phi::P
    theta::T
    denergy::DE
    dphi::DP
    dtheta::DT
end

EnergyAngleGrid(energy, phi, theta; denergy = nothing, dphi = nothing, dtheta = nothing) =
    EnergyAngleGrid(energy, phi, theta, denergy, dphi, dtheta)

"""
    plasma_moments(data, grid; species, units, sc_pot = 0, B = nothing, edim = 1)

Moments of one distribution `data` on `grid`.

- `data` is in `units`: `:eflux` [eV/(cm²·s·sr·eV)], `:flux` [1/(cm²·s·sr·eV)], `:df_km` [s³/km⁶] or `:df_cm` [s³/cm⁶].
- `species` is `:H`, `:He` (α), `:O` (O⁺), `:e`, or any object with `mass` [proton masses] and `charge` [e].
- `sc_pot` [V] shifts energies by `charge * sc_pot`.
- Output NamedTuple is CGS with energies in eV: `density` [cm⁻³], `velocity`, `vthermal` [cm/s], `flux` [cm⁻² s⁻¹],
  `eflux`, `qflux` [eV cm⁻² s⁻¹], `mftens`, `ptens` [eV cm⁻³] as `(xx, yy, zz, xy, xz, yz)`, `ttens`, `avgtemp`, `t3` [eV].
- With field `B`, adds `magt3` [eV] `(⊥, ⊥, ∥)` and the symmetry axis `symm` with `symm_theta`, `symm_phi`, `symm_ang` [deg].
"""
function plasma_moments(data, grid::EnergyAngleGrid; species, units, sc_pot = 0, B = nothing, edim = 1)
    (; mass, charge) = _species(species)
    coefficient, E_exp = _conversion_coeff(units, :eflux, mass)
    return _plasma_moments(data, grid, sc_pot, B; coefficient, E_exp, mass = mass * PROTON_MASS, charge, edim)
end

"""
    tmoments(data, grid; species, units, sc_pot = 0, B = nothing, tdim = ndims(data), edim = 1)::StructArray

Plasma moments of each slice of `data` along its time axis `tdim`.
`edim` indexes the axes of `data` including time. Align ancillary measurements to the scan intervals before calling.
`grid`, `sc_pot` and `B` are a vector with one value per time, or, for `grid` and `sc_pot`, one value for all times.
"""
function tmoments(data, grid; species, units, sc_pot = 0, B = nothing, tdim = ndims(data), edim = 1)
    1 <= tdim <= ndims(data) && 1 <= edim <= ndims(data) && tdim != edim ||
        throw(ArgumentError("tdim and edim must be distinct data axes"))
    nt = size(data, tdim)
    nt > 0 || throw(ArgumentError("tmoments needs at least one scan"))
    grid isa EnergyAngleGrid || _check_series(grid, nt, "grid")
    sc_pot isa Number || _check_series(sc_pot, nt, "sc_pot")
    if !isnothing(B)
        _check_series(B, nt, "B")
        all(b -> (b isa AbstractVector || b isa Tuple) && length(b) == 3, B) ||
            throw(DimensionMismatch("B needs one three-component field vector per scan"))
    end
    edim_slice = edim > tdim ? edim - 1 : edim
    moment_at(i) = plasma_moments(
        selectdim(data, tdim, i), _at(grid, i);
        sc_pot = _at(sc_pot, i), B = _at(B, i), edim = edim_slice, species, units
    )
    first_result = moment_at(1)
    result = StructArray{typeof(first_result)}(undef, nt)
    result[1] = first_result
    Threads.@threads :dynamic for i in 2:nt
        result[i] = moment_at(i)
    end
    return result
end

function _check_series(x, nt, name)
    x isa AbstractVector && axes(x, 1) == Base.OneTo(nt) ||
        throw(DimensionMismatch("$name needs one entry for each of $nt scans"))
end

_at(x, i) = x
_at(x::AbstractVector, i) = x[i]

_osize(data, dim) = ntuple(i -> size(data, i < dim ? i : i + 1), ndims(data) - 1)

_check_coordinate(::Union{Nothing, Real}, n, dims) = nothing
function _check_coordinate(x, n, dims)
    size(x) in ((n,), dims) || throw(DimensionMismatch("grid coordinate has shape $(size(x)); expected ($n,) or $dims"))
end

function _plasma_moments(data, grid, sc_pot, magf; coefficient, E_exp, mass, charge, edim)
    (; theta, phi) = grid
    _energy = grid.energy
    T = eltype(data)
    1 <= edim <= ndims(data) || throw(ArgumentError("edim must be a data axis"))
    ndims(data) in (2, 3) || throw(DimensionMismatch("expected one energy and one or two angle axes"))
    ne = size(data, edim)
    _check_coordinate(_energy, ne, size(data))
    _check_coordinate(grid.denergy, ne, size(data))
    edims = isnothing(grid.denergy) || ndims(grid.denergy) <= 1 ? size(_energy) : size(grid.denergy)
    eshape = ntuple(i -> i == edim ? ne : 1, ndims(data))
    energy_input = ndims(_energy) == 1 && length(edims) > 1 ? reshape(_energy, eshape) : _energy
    adims = _osize(data, edim)   # angle-only dimensions (everything after energy)
    _check_coordinate(phi, first(adims), adims)
    _check_coordinate(theta, last(adims), adims)
    _check_coordinate(grid.dphi, first(adims), adims)
    _check_coordinate(grid.dtheta, last(adims), adims)
    thetadims = size(theta)
    phidims = size(phi)
    ΔU = T(charge * sc_pot)

    return @no_escape begin
        Escale = @alloc(T, edims...)
        energy = @alloc(T, edims...)
        e_inf = @alloc(T, edims...)
        v_inf = @alloc(T, edims...)
        dtheta = @something(grid.dtheta, _axis_widths!(@alloc(T, thetadims...), theta, length(adims), nothing))
        dphi = @something(grid.dphi, _axis_widths!(@alloc(T, phidims...), phi, 1, 360))
        Omega = @alloc(T, 10, adims...) # Omega weights: angle-only (no energy axis)

        @. energy = _clean_energy(energy_input)
        dE = if isnothing(grid.denergy)
            _axis_widths!(@alloc(T, edims...), energy, edim, nothing)
        elseif ndims(grid.denergy) == 1 && length(edims) > 1
            reshape(grid.denergy, eshape)
        else
            grid.denergy
        end

        omega_weights!(Omega, theta, phi, dtheta, dphi)

        # Common derived arrays
        @. Escale = coefficient * dE * _weight(energy, T(dE), ΔU) * energy^(E_exp - 2)
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
