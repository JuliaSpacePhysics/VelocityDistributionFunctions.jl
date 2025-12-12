"""
    Maxwellian(vth, 𝐮₀)
    Maxwellian(T::Temperature, 𝐮₀; mass = me)

Isotropic Maxwellian velocity distribution with thermal velocity `vth` / temperature `T` and drift velocity `𝐮₀`.
"""
struct Maxwellian{T, VT} <: VelocityDistribution{T}
    vth::T
    u0::VT

    function Maxwellian(vth::T, u0::VT = _zero_𝐯(T); check_args = true) where {T, VT}
        @check_args Maxwellian (vth, vth >= zero(vth)) (u0, length(u0) == 3)
        return new{T, VT}(vth, u0)
    end
end

function _rand!(rng::AbstractRNG, d::Maxwellian, 𝐯::AbstractVector)
    return @. 𝐯 = d.vth / sqrt(2) * SA[randn(rng), randn(rng), randn(rng)] + d.u0
end

# Generalal pdf that supports unitful inputs
function _pdf(d::Maxwellian, 𝐯)
    return sqrt(π^-3) * d.vth^-3 * exp(-sqdist(d.u0, 𝐯) / d.vth^2)
end
