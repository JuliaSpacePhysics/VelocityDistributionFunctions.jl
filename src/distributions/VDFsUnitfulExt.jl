using Unitful: upreferred, @derived_dimension
using Unitful: Quantity, Temperature, Velocity, Pressure
using Unitful: me, k
import Unitful: ustrip
using ConstructionBase: constructorof, getfields
using Unitful: 𝐋
@derived_dimension NumberDensity 𝐋^-3


v_th(T, m) = upreferred(sqrt(2 * k * T / m))

const vth_kappa_ = kappa_thermal_speed

"""
    ustrip(d::VelocityDistribution)

Strip units from all fields of a velocity distribution, returning a new distribution
with unitless values.
"""
function ustrip(d::T) where {T <: AbstractVelocityPDF}
    fields = map(x -> ustrip.(x), getfields(d))
    return constructorof(T)(fields...; check_args = false)
end

function ustrip(d::VelocityDistribution)
    return VelocityDistribution(ustrip(d.shape), ustrip(d.n))
end


Maxwellian(T::Temperature, args...; mass = me, kw...) =
    Maxwellian(v_th(T, mass), args...; kw...)

BiMaxwellian(T_perp::Temperature, T_para::Temperature, args...; mass = me, kw...) =
    BiMaxwellian(v_th(T_perp, mass), v_th(T_para, mass), args...; kw...)

Kappa(T_perp::Temperature, κ, args...; mass = me, kw...) =
    Kappa(vth_kappa_(T_perp, κ, mass), κ, args...; kw...)

BiKappa(T_perp::Temperature, T_para::Temperature, κ, args...; mass = me, kw...) =
    BiKappa(vth_kappa_(T_perp, κ, mass), vth_kappa_(T_para, κ, mass), κ, args...; kw...)

Distributions.pdf(d::AbstractVelocityPDF, v::AbstractVector{<:Velocity}) = _pdf(d, v)

for f in (:Maxwellian, :BiMaxwellian, :Kappa, :BiKappa)
    @eval $f(n::NumberDensity, T::Temperature, args...; kw...) = VelocityDistribution(n, $f(T, args...; kw...))
end

for f in (:Maxwellian, :Kappa)
    @eval $f(n::NumberDensity, p::Pressure, args...; kw...) = VelocityDistribution(n, $f(p / (n * k), args...; kw...))
end

for f in (:BiMaxwellian, :BiKappa)
    @eval $f(n::NumberDensity, p_perp::Pressure, p_para::Pressure, args...; kw...) =
        VelocityDistribution(n, $f(p_perp / (n * k), p_para / (n * k), args...; kw...))
end
