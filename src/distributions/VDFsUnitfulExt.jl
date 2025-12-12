using Unitful: upreferred
using Unitful: Quantity, Temperature, Velocity
using Unitful: me, k
import Unitful: ustrip
using ConstructionBase: constructorof, getfields

v_th(T, m) = upreferred(sqrt(2 * k * T / m))

const vth_kappa_ = kappa_thermal_speed

"""
    ustrip(d::VelocityDistribution)

Strip units from all fields of a velocity distribution, returning a new distribution
with unitless values.
"""
function ustrip(d::T) where {T <: VelocityDistribution}
    props = getfields(d)
    stripped_props = map(ustrip, props)
    return constructorof(T)(stripped_props...; check_args = false)
end


Maxwellian(T::Temperature, args...; mass = me, kw...) =
    Maxwellian(v_th(T, mass), args...; kw...)

BiMaxwellian(T_perp::Temperature, T_para::Temperature, args...; mass = me, kw...) =
    BiMaxwellian(v_th(T_perp, mass), v_th(T_para, mass), args...; kw...)

Kappa(T_perp::Temperature, κ, args...; mass = me, kw...) =
    Kappa(vth_kappa_(T_perp, κ, mass), κ, args...; kw...)

Distributions.pdf(d::VelocityDistribution, v::AbstractVector{<:Velocity}) = _pdf(d, v)

function Random.rand(d::VelocityDistribution{<:Quantity})
    v_sample = similar(d.u0)
    _rand!(Random.default_rng(), d, v_sample)
    return v_sample
end

# TODO: support multiple dimensions
Random.rand(d::VelocityDistribution{<:Quantity}, dim::Int) = [rand(d) for _ in 1:dim]
