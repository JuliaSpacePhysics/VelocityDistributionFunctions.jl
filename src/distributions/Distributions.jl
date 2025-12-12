using Random
using Distributions
using Distributions: MultivariateDistribution, @check_args
import Distributions: pdf
using LinearAlgebra: normalize, dot
using SpecialFunctions: gamma
using BaseType: base_numeric_type

import Random: rand

"""
Abstract type for velocity distribution functions, extending MultivariateDistribution.
"""
abstract type VelocityDistribution{T} <: MultivariateDistribution{Distributions.Continuous} end

(d::VelocityDistribution)(𝐯) = pdf(d, 𝐯)

Base.broadcastable(x::VelocityDistribution) = Ref(x)
Random.rand(d::VelocityDistribution, dim::Int) = rand(d, (dim,))

# ---
# Distributions interface
Base.length(::VelocityDistribution) = 3
# Handle method ambiguity (`Distributions` assume Real type output)
Distributions._rand!(rng::AbstractRNG, d::VelocityDistribution, x::AbstractVector{<:Real}) = _rand!(rng, d, x)
Distributions.pdf(d::VelocityDistribution, 𝐯::AbstractVector{<:Real}) = _pdf(d, 𝐯)

# ---
# Internal utilities
sqdist(v, u) = sum(abs2, v - u)
_zero_𝐯(T) = zero(SVector{3, T})

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

include("Maxwellian.jl")
include("BiMaxwellian.jl")
include("Kappa.jl")
include("BiKappa.jl")
include("VDFsUnitfulExt.jl")

function Distributions.pdf(vdf::Union{Maxwellian, Kappa}, v::V)
    return 4π * v.val^2 * _pdf(vdf, SA[v.val, 0, 0])
end
