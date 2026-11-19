const _SPECIES = (H = Proton, He = Particle("{4}He", 2), O = Particle("{16}O", 1), e = Electron)

# mass [eV/(cm/s)²], charge [e]
function _species(s::Symbol)
    haskey(_SPECIES, s) || throw(ArgumentError("species must be one of $(keys(_SPECIES)) or a `PlasmaPopulations.Particle`; got :$s"))
    return _species(_SPECIES[s])
end
_species(p) = (; mass = p.m * 1.0e-4 / E_SI, charge = p.q / E_SI)

# (c, p) with `out = data * c * E^p`, for particle mass `m` [eV/(cm/s)²].
function _conversion_coeff(from::Symbol, to::Symbol, m)
    c_from, p_from = _to_df_cm(from, m)
    c_to, p_to = _to_df_cm(to, m)
    return (c_from / c_to, p_from - p_to)
end

# (c, p) with `f[s³/cm⁶] = data * c * E^p`, from `F = 2E/m² · f` and `j = E · F`.
function _to_df_cm(units::Symbol, m)
    units === :df_cm && return (1.0, 0)
    units === :df_km && return (1.0e-30, 0)
    units === :flux && return (m^2 / 2, -1)
    units === :eflux && return (m^2 / 2, -2)
    throw(ArgumentError("units must be one of :flux, :eflux, :df_km, :df_cm; got $(repr(units))"))
end
