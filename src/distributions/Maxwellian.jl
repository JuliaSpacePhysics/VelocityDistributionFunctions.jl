"""
    Maxwellian(vth, 𝐮₀)
    Maxwellian(T::Temperature, 𝐮₀; mass = me)

Isotropic Maxwellian velocity distribution with thermal velocity `vth` / temperature `T` and drift velocity `𝐮₀`.
"""
struct MaxwellianPDF{T, VT} <: AbstractVelocityPDF
    vth::T
    u0::VT

    function MaxwellianPDF(vth::T, u0::VT = _zero_𝐯(T); check_args = true) where {T, VT}
        @check_args MaxwellianPDF (vth, vth >= zero(vth)) (u0, length(u0) == 3)
        return new{T, VT}(vth, u0)
    end
end

Maxwellian(args...; kw...) = MaxwellianPDF(args...; kw...)

function _rand!(rng::AbstractRNG, d::MaxwellianPDF, 𝐯::AbstractVector)
    return @. 𝐯 = d.vth / sqrt(2) * SA[randn(rng), randn(rng), randn(rng)] + d.u0
end

# Generalal pdf that supports unitful inputs
function _pdf(d::MaxwellianPDF, 𝐯)
    return sqrt(π^-3) * d.vth^-3 * exp(-sqdist(d.u0, 𝐯) / d.vth^2)
end
