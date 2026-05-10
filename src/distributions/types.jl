"""
Abstract type for velocity probability distribution functions.
"""
abstract type AbstractVelocityPDF{T} end

(d::AbstractVelocityPDF)(𝐯) = pdf(d, 𝐯)
pdf(d::AbstractVelocityPDF, 𝐯) = _pdf(d, 𝐯)
Base.eltype(::Type{<:AbstractVelocityPDF{T}}) where {T} = T
Base.broadcastable(x::AbstractVelocityPDF) = Ref(x)

"""
Abstract type for velocity distributions.
"""
abstract type AbstractVelocityDistribution end

Base.broadcastable(x::AbstractVelocityDistribution) = Ref(x)

"""
    VelocityDistribution(n, pdf)

Physical velocity distribution combining number density `n` with a velocity probability distribution `pdf`.
"""
struct VelocityDistribution{N, D} <: AbstractVelocityDistribution
    n::N
    pdf::D

    function VelocityDistribution(n::N, pdf::D; check_args = true) where {N, D}
        @check_args VelocityDistribution (n, n >= zero(n))
        return new{N, D}(n, pdf)
    end
end

(d::VelocityDistribution)(𝐯) = d.n * pdf(d.pdf, 𝐯)

function Base.getproperty(d::VelocityDistribution, sym::Symbol)
    return sym in fieldnames(VelocityDistribution) ? getfield(d, sym) : getproperty(d.pdf, sym)
end

function Random.rand(d::AbstractVelocityPDF)
    v_sample = MVector{length(d), eltype(d)}(undef)
    _rand!(Random.default_rng(), d, v_sample)
    return v_sample
end

# TODO: support multiple dimensions
Random.rand(d::AbstractVelocityPDF, dim::Int) = [rand(d) for _ in 1:dim]
Random.rand(rng::AbstractRNG, X::VelocityDistribution, dims::Dims) = rand(rng, X.pdf, dims)
Random.rand(rng::AbstractRNG, d::VelocityDistribution) = rand(rng, d.pdf)
Random.rand(d::VelocityDistribution, args::Integer...) = rand(d.pdf, args...)

Base.length(::AbstractVelocityPDF) = 3
