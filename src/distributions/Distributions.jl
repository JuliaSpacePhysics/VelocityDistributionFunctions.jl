using Random
using LinearAlgebra: normalize, dot
using SpecialFunctions: gamma
using BaseType: base_numeric_type

import Random: rand

"""
    pdf(d, 𝐯)

Evaluate the probability density function of distribution `d` at velocity `𝐯`.
"""
function pdf end

macro check_args(D, args...)
    checks = map(args) do arg
        var, cond = arg.args
        msg = "$(D): the condition $(cond) is not satisfied."
        :($(cond) || throw(DomainError($(var), $msg)))
    end
    return esc(quote
        if check_args
            $(checks...)
        end
    end)
end

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

function pdf(vdf::Union{MaxwellianPDF, KappaPDF}, v::V)
    v² = v.val^2
    return 4π * v² * _pdf_v²(vdf, v²)
end

function pdf(d::ShiftedPDF, v::VPar)
    upar = d.u0 ⋅ d.base.b0
    return pdf(d.base, VPar(v.val - upar))
end
