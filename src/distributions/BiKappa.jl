"""
    BiKappa(vth_perp, vth_para, κ, 𝐮₀=[0, 0, 0], b0=[0, 0, 1])
    BiKappa(T_perp::Temperature, T_para::Temperature, κ, 𝐮₀=[0, 0, 0], b0=[0, 0, 1]; mass = me)

BiKappa velocity distribution with kappa index `κ`, different thermal velocities in perpendicular
`vth_perp` and parallel `vth_para` directions, drift velocity `𝐮₀` and magnetic field direction `b0`.

```math
f(𝐯) ∝ \\left[1 + \\frac{(𝐯_⟂ - 𝐮_{0, ⟂})^{2}/v_{\\mathrm{th}, ⟂}^{2} + (𝐯_∥ - 𝐮_{0, ∥})^{2}/v_{\\mathrm{th}, ∥}^{2}}{κ}\\right]^{-(κ+1)}
```

where the normalization constant is
``A = Γ(κ + 1) / Γ(κ - 1/2) / ((π κ)^{3/2} v_{th,∥} v_{th,⟂}^2)``.
"""
struct BiKappa{T, K <: Real, TB, TVD} <: VelocityDistribution{T}
    vth_perp::T
    vth_para::T
    κ::K
    b0::TB
    u0::TVD

    function BiKappa(
            vth_perp::T, vth_para::T, κ::K,
            u0::TVD = _zero_𝐯(T), b0::TB = SA[0.0, 0.0, 1.0];
            check_args = true
        ) where {T, K, TVD, TB}
        @check_args BiKappa (κ, κ > 1.5) (vth_perp, vth_perp > zero(vth_perp)) (vth_para, vth_para > zero(vth_para)) (b0, length(b0) == 3) (u0, length(u0) == 3)
        BT = base_numeric_type(T)
        B_normalized = normalize(BT.(b0))
        return new{T, K, TB, TVD}(vth_perp, vth_para, κ, B_normalized, u0)
    end
end

_Aκ_bi(κ, vth_perp, vth_para) = gamma(κ + 1) / gamma(κ - 1 / 2) / √((π * κ)^3) / (vth_para * vth_perp^2)

function _pdf(d::BiKappa, 𝐯)
    d𝐯 = 𝐯 - d.u0
    dv_para = d𝐯 ⋅ d.b0
    v_perp_sq = sum(abs2, d𝐯 - dv_para * d.b0)
    w² = (dv_para^2 / d.vth_para^2 + v_perp_sq / d.vth_perp^2) / d.κ
    expTerm = (1 + w²)^(-(d.κ + 1))
    return _Aκ_bi(d.κ, d.vth_perp, d.vth_para) * expTerm
end

function _rand!(rng::AbstractRNG, d::BiKappa{T}, x) where {T}
    bperp1 = normalize(d.b0 × get_least_parallel_basis_vector(d.b0))
    bperp2 = d.b0 × bperp1

    ν = 2 * d.κ - 1
    ξ = rand(rng, Chisq(ν))
    scale = sqrt(d.κ / ξ)

    vpara = d.vth_para * scale * randn(rng)
    vperp_1 = d.vth_perp * scale * randn(rng)
    vperp_2 = d.vth_perp * scale * randn(rng)

    @. x = d.u0 + vpara * d.b0 + vperp_1 * bperp1 + vperp_2 * bperp2
    return x
end
