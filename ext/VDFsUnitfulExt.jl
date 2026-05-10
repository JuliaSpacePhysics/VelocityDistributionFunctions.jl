module VDFsUnitfulExt

using VelocityDistributionFunctions
using VelocityDistributionFunctions: AbstractVelocityPDF, VelocityDistribution
using VelocityDistributionFunctions: _pdf
import VelocityDistributionFunctions: MaxwellianPDF, BiMaxwellianPDF, KappaPDF, BiKappaPDF
import VelocityDistributionFunctions: Maxwellian, BiMaxwellian, Kappa, BiKappa
import VelocityDistributionFunctions: kappa_thermal_speed
using Unitful: upreferred, @derived_dimension
using Unitful: Quantity, Temperature, Velocity, Pressure
using Unitful: me, k
import Unitful: ustrip
using ConstructionBase: constructorof, getfields
using Unitful: 𝐋

@derived_dimension NumberDensity 𝐋^-3
const NType = Union{NumberDensity, Real}

v_th(T, m) = upreferred(sqrt(2 * k * T / m))

kappa_thermal_speed(T::Temperature, κ, m) = kappa_thermal_speed(v_th(T, m), κ)

"""
    ustrip(d::VelocityDistribution)

Strip units from all fields of a velocity distribution, returning a new distribution
with unitless values.
"""
function ustrip(d::T) where {T <: Union{AbstractVelocityPDF, VelocityDistribution}}
    fields = map(x -> ustrip.(x), getfields(d))
    return constructorof(T)(fields...; check_args = false)
end

MaxwellianPDF(T::Temperature; mass = me, kw...) =
    MaxwellianPDF(v_th(T, mass); kw...)

BiMaxwellianPDF(T_perp::Temperature, T_para::Temperature, args...; mass = me, kw...) =
    BiMaxwellianPDF(v_th(T_perp, mass), v_th(T_para, mass), args...; kw...)

KappaPDF(T_perp::Temperature, κ; mass = me, kw...) =
    KappaPDF(kappa_thermal_speed(T_perp, κ, mass), κ; kw...)

BiKappaPDF(T_perp::Temperature, T_para::Temperature, κ, args...; mass = me, kw...) =
    BiKappaPDF(kappa_thermal_speed(T_perp, κ, mass), kappa_thermal_speed(T_para, κ, mass), κ, args...; kw...)

import VelocityDistributionFunctions: pdf
pdf(d::AbstractVelocityPDF, v::AbstractVector{<:Velocity}) = _pdf(d, v)

for f in (:Maxwellian, :BiMaxwellian, :Kappa, :BiKappa)
    @eval $f(n::NType, T::Temperature, args...; kw...) = VelocityDistribution(n, $f(T, args...; kw...))
end

for f in (:Maxwellian, :Kappa)
    @eval $f(n::NumberDensity, p::Pressure, args...; kw...) = VelocityDistribution(n, $f(p / (n * k), args...; kw...))
end

for f in (:BiMaxwellian, :BiKappa)
    @eval $f(n::NumberDensity, p_perp::Pressure, p_para::Pressure, args...; kw...) =
        VelocityDistribution(n, $f(p_perp / (n * k), p_para / (n * k), args...; kw...))
end

end
