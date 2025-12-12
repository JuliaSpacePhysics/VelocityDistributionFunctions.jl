using Random
using Distributions
using Distributions: MultivariateDistribution, @check_args
import Distributions: pdf
using LinearAlgebra: normalize, dot
using SpecialFunctions: gamma
using BaseType: base_numeric_type

import Random: rand

# Speed
struct V{T}
    val::T
end
struct VPar{T}
    val::T
end
struct VPerp{T}
    val::T
end

include("types.jl")
include("ShiftedPDF.jl")
include("Maxwellian.jl")
include("BiMaxwellian.jl")
include("Kappa.jl")
include("BiKappa.jl")

for (f, g) in [(:Maxwellian, :MaxwellianPDF), (:BiMaxwellian, :BiMaxwellianPDF), (:Kappa, :KappaPDF), (:BiKappa, :BiKappaPDF)]

    doc = """
        $f(args...; u0=nothing, kw...)
        $f(n, args...; u0=nothing, kw...)

    Construct a [`$(g)`](@ref) velocity distribution.

    If `n` is provided, returns [`VelocityDistribution`](@ref) with `n` and the distribution `$(g)(args...; kw...)`.
    If `u0` is provided, returns [`ShiftedPDF`](@ref) with the distribution `$(g)(args...; kw...)` and `u0`.
    """

    @eval @doc $doc function $f(args...; u0 = nothing, kw...)
        return if length(args) == fieldcount($g) + 1
            VelocityDistribution(args[1], $f(args[2:end]...; u0, kw...))
        else
            base = $g(args...; kw...)
            isnothing(u0) ? base : ShiftedPDF(base, u0)
        end
    end
end

include("VDFsUnitfulExt.jl")

function Distributions.pdf(vdf::Union{MaxwellianPDF, KappaPDF}, v::V)
    return 4π * v.val^2 * _pdf(vdf, SA[v.val, 0, 0])
end

function Distributions.pdf(d::ShiftedPDF, v::VPar)
    upar = d.u0 ⋅ d.base.b0
    return pdf(d.base, VPar(v.val - upar))
end
