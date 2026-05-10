module VDFsDistributionsExt

using VelocityDistributionFunctions
using VelocityDistributionFunctions: AbstractVelocityPDF, _rand!
import VelocityDistributionFunctions: pdf
import Distributions
using Random: AbstractRNG

Distributions.pdf(d::AbstractVelocityPDF, 𝐯::AbstractVector) = pdf(d, 𝐯)
Distributions._rand!(rng::AbstractRNG, d::AbstractVelocityPDF, x::AbstractVector{<:Real}) = _rand!(rng, d, x)

end
