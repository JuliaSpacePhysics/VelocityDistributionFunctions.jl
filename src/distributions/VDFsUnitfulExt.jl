using Unitful: upreferred
using Unitful: Quantity, Temperature, Velocity
using Unitful: me, k

v_th(T, m) = upreferred(sqrt(2 * k * T / m))

const vth_kappa_ = kappa_thermal_speed

Maxwellian(T::Temperature, args...; mass = me, kw...) =
    Maxwellian(v_th(T, mass), args...; kw...)

BiMaxwellian(T_perp::Temperature, T_para::Temperature, args...; mass = me, kw...) =
    BiMaxwellian(v_th(T_perp, mass), v_th(T_para, mass), args...; kw...)

Kappa(T_perp::Temperature, κ, args...; mass = me, kw...) =
    Kappa(vth_kappa_(T_perp, κ, mass), κ, args...; kw...)

Distributions.pdf(d::VelocityDistribution, v::AbstractVector{<:Velocity}) = _pdf(d, v)

function Random.rand(d::VelocityDistribution{<:Quantity})
    v_sample = similar(d.u0)
    _rand!(Random.default_rng(), d, v_sample)
    return v_sample
end

# TODO: support multiple dimensions
Random.rand(d::VelocityDistribution{<:Quantity}, dim::Int) = [rand(d) for _ in 1:dim]
