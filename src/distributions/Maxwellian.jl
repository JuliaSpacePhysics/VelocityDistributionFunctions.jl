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
    return @. 𝐯 = d.vth / sqrt(2) * (randn(rng), randn(rng), randn(rng))
end

# Generalal pdf that supports unitful inputs
_pdf(d::MaxwellianPDF, 𝐯) = _pdf_v²(d, sum(abs2, 𝐯))

function _pdf_v²(d::MaxwellianPDF, v²)
    return sqrt(π^-3) * d.vth^-3 * exp(-v² / d.vth^2)
end