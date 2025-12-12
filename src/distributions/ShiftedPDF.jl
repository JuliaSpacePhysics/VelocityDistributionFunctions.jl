"""
    ShiftedPDF(base, u0)

Velocity distribution with a drift velocity `u0` applied to a `base` distribution.
"""
struct ShiftedPDF{T, D, U} <: AbstractVelocityPDF{T}
    base::D
    u0::U
end

function ShiftedPDF(base::AbstractVelocityPDF{T}, u0; check_args = true) where {T}
    @check_args ShiftedPDF (u0, length(u0) == length(base))
    return ShiftedPDF{T, typeof(base), typeof(u0)}(base, u0)
end

function _pdf(d::ShiftedPDF, 𝐯)
    return pdf(d.base, 𝐯 .- d.u0)
end

function _rand!(rng::AbstractRNG, d::ShiftedPDF, x)
    _rand!(rng, d.base, x)
    return x .+= d.u0
end
