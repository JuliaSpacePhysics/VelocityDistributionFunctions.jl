# https://github.com/spedas/pyspedas/blob/master/pyspedas/projects/mms/particles/mms_convert_flux_units.py

const PROTON_MASS = 0.0104535 * 1.0e-10  # eV/(cm/s)²

# Species lookup: mass in eV/(cm/s)², charge in elementary charges, A for unit conversion
const _SPECIES_TABLE = Dict(
    :H => (mass = 1.0 * PROTON_MASS, charge = 1.0, A = 1.0),
    :He => (mass = 4.0 * PROTON_MASS, charge = 2.0, A = 4.0),
    :O => (mass = 16.0 * PROTON_MASS, charge = 1.0, A = 16.0),
    :e => (mass = PROTON_MASS / 1836.0, charge = -1.0, A = 1.0 / 1836.0),
)

# Supported species: `:H` (proton), `:He` (alpha), `:O` (oxygen), `:e` (electron).
species_info(s::Symbol) = _SPECIES_TABLE[s]

"""
    

Returns (coefficient, energy_exponent) such that: out = data * c * E^p

Conversion chain: `df_cm ↔ df_km ↔ flux ↔ eflux`

- `f [s³/cm⁶]` — phase space density (CGS)
- `f [s³/km⁶]` — phase space density (SI-ish)
- `F [#/(cm²·s·sr·eV)]` — differential number flux, `F = 2E/m² · f`
- `j [eV/(cm²·s·sr·eV)]` — differential energy flux, `j = E · F`
"""
@inline function _conversion_coeff(from, to, A)
    from == to && return (1.0, 0)
    flux_to_df = A^2 * 0.5447e6
    c_in, p_in = _to_flux_coeff(Val(from), flux_to_df)
    c_out, p_out = _from_flux_coeff(Val(to), flux_to_df)
    return (c_in * c_out, p_in + p_out)
end

# any unit → flux: (coefficient, energy_exponent)
_to_flux_coeff(::Val{:flux}, c) = (1.0, 0)
_to_flux_coeff(::Val{:eflux}, c) = (1.0, -1)
_to_flux_coeff(::Val{:df_km}, c) = (1.0 / c, 1)
_to_flux_coeff(::Val{:df_cm}, c) = (1.0e30 / c, 1)

# flux → any unit: (coefficient, energy_exponent)
_from_flux_coeff(::Val{:flux}, c) = (1.0, 0)
_from_flux_coeff(::Val{:eflux}, c) = (1.0, 1)
_from_flux_coeff(::Val{:df_km}, c) = (c, -1)
_from_flux_coeff(::Val{:df_cm}, c) = (c / 1.0e30, -1)
