module VDFsDistributionsExt

import Distributions
import VelocityDistributionFunctions: AbstractVelocityPDF, pdf

Distributions.pdf(d::AbstractVelocityPDF, v::AbstractVector) = pdf(d, v)

end
