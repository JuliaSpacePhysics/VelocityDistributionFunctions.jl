const _SPECIES = (H = Proton, He = Particle("{4}He", 2), O = Particle("{16}O", 1), e = Electron)

# mass [kg], charge [C]
function _species(s::Symbol)
    haskey(_SPECIES, s) || throw(ArgumentError("species must be one of $(keys(_SPECIES)) or a `PlasmaPopulations.Particle`; got :$s"))
    return _species(_SPECIES[s])
end
_species(p) = (; mass = p.m, charge = p.q)

# kg → eV/(cm/s)², the mass unit matching the energies and lengths of `units`
_mass_eV_cm(m) = m * 1.0e-4 / E_SI

function convert_flux_units(data, energy, units::Pair{Symbol, Symbol}; species = nothing)
    mass = isnothing(species) ? nothing : _species(species).mass
    c, p = _conversion_coeff(units.first, units.second, mass)
    return @. data * c * _ipow(energy, p)
end

convert_flux_units(data, grid::EnergyAngleBins, units::Pair{Symbol, Symbol}; kw...) =
    convert_flux_units(data, _energies(grid), units; kw...)

# E^p for the p ∈ -2:2 that conversions produce; generic `^(::Float64, ::Int)` is way slower
_ipow(E, p) = p == -2 ? inv(E * E) : p == -1 ? inv(E) : p == 0 ? one(E) : p == 1 ? E : p == 2 ? E * E : E^p

# (c, p) with `out = data * c * E^p`, for particle mass `m` [kg]
function _conversion_coeff(from::Symbol, to::Symbol, m)
    c_from, p_from, k_from = _to_df_cm(from)
    c_to, p_to, k_to = _to_df_cm(to)
    k = k_from - k_to
    k == 0 || !isnothing(m) || throw(ArgumentError("converting :$from to :$to needs `species`"))
    return (c_from / c_to * (k == 0 ? 1.0 : _mass_eV_cm(m)^k), p_from - p_to)
end

# (c, p, k) with `f[s³/cm⁶] = data * c * E^p * m^k`, from `F = 2E/m² · f` and `j = E · F`.
function _to_df_cm(units::Symbol)
    units === :df_cm && return (1.0, 0, 0)
    units === :df_km && return (1.0e-30, 0, 0)
    units === :df_m && return (1.0e-12, 0, 0)
    units === :flux && return (0.5, -1, 2)
    units === :eflux && return (0.5, -2, 2)
    throw(ArgumentError("units must be one of :flux, :eflux, :df_cm, :df_km, :df_m; got $(repr(units))"))
end
