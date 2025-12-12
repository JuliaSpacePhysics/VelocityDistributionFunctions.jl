"""
    BiKappaPDF(vth_perp, vth_para, κ, b0=[0, 0, 1])
    BiKappaPDF(T_perp::Temperature, T_para::Temperature, κ, b0=[0, 0, 1]; mass = me)

Modified BiKappa velocity distribution with kappa index `κ`, assuming κ-independent temperatures ``T_{∥,⟂}``, with magnetic field direction `b0`. 

The distribution can also be parameterized by kappa thermal speeds ``v_{th,∥}`` and ``v_{th,⟂}``.

```math
\\begin{aligned}
f_κ(𝐯) & ∝ \\left[1 + \\frac{v_∥^2}{κ v_{\\mathrm{th}, ∥}^2} + \\frac{v_⟂^2}{κ v_{\\mathrm{th}, ⟂}^2}\\right]^{- κ - 1} \\\\
    & = \\left[1+\\frac{m}{k_B (2 κ-3)} \\left(\\frac{v_∥^2}{T_∥}+\\frac{v_⟂^2}{T_⟂}\\right) \\right]^{-κ-1}
\\end{aligned}
```

where the normalization constant is

```math
\\begin{aligned}
A_κ &= \\left(\\frac{1}{π κ}\\right)^{3/2} \\frac{1}{v_{th,∥} v_{th,⟂}^2} \\frac{Γ[κ+1]}{Γ[κ-1/2]} \\\\
  &= \\left[\\frac{m}{π k_B(2 κ-3)}\\right]^{3 / 2} \\frac{1}{T_⟂ \\sqrt{T_∥}} \\frac{Γ[κ+1]}{Γ[κ-1/2]}
\\end{aligned}
```

See also [`Kappa`](@ref), [`kappa_thermal_speed`](@ref)
"""
struct BiKappaPDF{T, K <: Real, TB} <: KappaDistribution{T, K}
    vth_perp::T
    vth_para::T
    κ::K
    b0::TB

    function BiKappaPDF(
            vth_perp::T, vth_para::T, κ::K,
            b0::TB = SA[0.0, 0.0, 1.0];
            check_args = true
        ) where {T, K, TB}
        @check_args BiKappaPDF (κ, κ > 1.5) (vth_perp, vth_perp > zero(vth_perp)) (vth_para, vth_para > zero(vth_para)) (b0, length(b0) == 3)
        BT = base_numeric_type(T)
        B_normalized = normalize(BT.(b0))
        return new{T, K, TB}(vth_perp, vth_para, κ, B_normalized)
    end
end

_Aκ_bi(κ, vth_perp, vth_para) = gamma(κ + 1) / gamma(κ - 1 / 2) / √((π * κ)^3) / (vth_para * vth_perp^2)

function _pdf(d::BiKappaPDF, 𝐯)
    dv_para = 𝐯 ⋅ d.b0
    v_perp_sq = sum(abs2, 𝐯 - dv_para * d.b0)
    w² = (dv_para^2 / d.vth_para^2 + v_perp_sq / d.vth_perp^2) / d.κ
    expTerm = (1 + w²)^(-(d.κ + 1))
    return _Aκ_bi(d.κ, d.vth_perp, d.vth_para) * expTerm
end

function _rand!(rng::AbstractRNG, d::BiKappaPDF{T}, x) where {T}
    bperp1 = normalize(d.b0 × get_least_parallel_basis_vector(d.b0))
    bperp2 = d.b0 × bperp1

    ν = 2 * d.κ - 1
    ξ = rand(rng, Chisq(ν))
    scale = sqrt(d.κ / ξ)

    vpara = d.vth_para * scale * randn(rng)
    vperp_1 = d.vth_perp * scale * randn(rng)
    vperp_2 = d.vth_perp * scale * randn(rng)

    return @. x = vpara * d.b0 + vperp_1 * bperp1 + vperp_2 * bperp2
end
