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
abstract type AbstractVelocityPDF <: MultivariateDistribution{Distributions.Continuous} end

abstract type AbstractVelocityDistribution end

struct VelocityDistribution{N, D} <: AbstractVelocityDistribution
    n::N
    shape::D
end

(d::AbstractVelocityPDF)(𝐯) = pdf(d, 𝐯)
(d::VelocityDistribution)(𝐯) = d.n * pdf(d.shape, 𝐯)

Distributions.pdf(d::VelocityDistribution, 𝐯) = d.n * pdf(d.shape, 𝐯)

Base.broadcastable(x::AbstractVelocityPDF) = Ref(x)
Base.broadcastable(x::VelocityDistribution) = Ref(x)

function Random.rand(d::AbstractVelocityPDF)
    v_sample = similar(d.u0)
    _rand!(Random.default_rng(), d, v_sample)
    return v_sample
end

# TODO: support multiple dimensions
Random.rand(d::AbstractVelocityPDF, dim::Int) = [rand(d) for _ in 1:dim]
Random.rand(d::VelocityDistribution, dim::Int) = [rand(d.shape) for _ in 1:dim]

# ---
# Distributions interface
Base.length(::AbstractVelocityPDF) = 3
Base.length(d::VelocityDistribution) = length(d.shape)
# Handle method ambiguity (`Distributions` assume Real type output)
Distributions._rand!(rng::AbstractRNG, d::AbstractVelocityPDF, x::AbstractVector{<:Real}) = _rand!(rng, d, x)
Distributions.pdf(d::AbstractVelocityPDF, 𝐯::AbstractVector{<:Real}) = _pdf(d, 𝐯)

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

function Distributions.pdf(vdf::Union{MaxwellianPDF, KappaPDF}, v::V)
    return 4π * v.val^2 * _pdf(vdf, SA[v.val, 0, 0])
end
