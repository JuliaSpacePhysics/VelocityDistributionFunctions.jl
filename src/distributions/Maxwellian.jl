"""
    Maxwellian(vth)
    Maxwellian(T::Temperature; mass = me)

Isotropic Maxwellian velocity distribution with thermal velocity `vth` / temperature `T`.
"""
struct MaxwellianPDF{T} <: AbstractVelocityPDF{T}
    vth::T

    function MaxwellianPDF(vth::T; check_args = true) where {T}
        @check_args MaxwellianPDF (vth, vth >= zero(vth))
        return new{T}(vth)
    end
end

function _rand!(rng::AbstractRNG, d::MaxwellianPDF, 𝐯::AbstractVector)
    return @. 𝐯 = d.vth / sqrt(2) * SA[randn(rng), randn(rng), randn(rng)]
end

# Generalal pdf that supports unitful inputs
function _pdf(d::MaxwellianPDF, 𝐯)
    return sqrt(π^-3) * d.vth^-3 * exp(-sum(abs2, 𝐯) / d.vth^2)
end
