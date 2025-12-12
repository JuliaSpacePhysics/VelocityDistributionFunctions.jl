"""
    Kappa(κ, vth, 𝐮₀=[0, 0, 0])

Kappa velocity distribution with index `κ` and thermal velocity `vth`, with optional drift velocity `𝐮₀`.

```math
f(𝐯) ∝ [1 + |𝐯 - 𝐮₀|²/(κ·vₜₕ²)]^{-(κ+1)}
```

where the normalization constant is ``A_3 = Γ(κ + 1) / (π κ v_{th}^2)^{3/2} / Γ(κ - 1/2)``.

# Notes
Kappa index must be > 1.5 for finite variance. For large κ, the distribution approaches a Maxwellian. Smaller κ values produce stronger high-energy tails.

See also [pierrardSuprathermalPopulationsTheir2021](@citet) and [pierrardKappaDistributionsTheory2010](@citet).
"""
struct Kappa{T, K <: Real, U} <: VelocityDistribution{T}
    vth::T
    κ::K
    u0::U

    function Kappa(vth::T, κ::K, u0::U = _zero_𝐯(T); check_args = true) where {T, K, U}
        @check_args Kappa (κ, κ > 1.5) (vth, vth > zero(vth)) (u0, length(u0) == 3)
        return new{T, K, U}(vth, κ, u0)
    end
end

"""
    kappa_thermal_speed(T, κ, m)

Return the most probable speed of a (modified) kappa distribution with κ-Independent temperature `T`.

```math
V_{th,i} = \\sqrt{\\frac{κ - 3/2}{κ} \\frac{2 k_B T}{m}}
```
"""
function kappa_thermal_speed(T, κ, m)
    return upreferred(sqrt(2 * k * T / m)) * sqrt((κ - 3 / 2) / κ)
end


_Aκ(κ, vth) = gamma(κ + 1) / gamma(κ - 1 / 2) / √((π * κ)^3) / vth^3

function _pdf(d::Kappa, 𝐯)
    w² = sqdist(𝐯, d.u0) / (d.κ * d.vth^2)
    expTerm = (1 + w²)^(-(d.κ + 1))
    return _Aκ(d.κ, d.vth) * expTerm
end

function _pdf_1d(d::Kappa, vx)
    w² = (vx - d.u0[1])^2 / (d.κ * d.vth^2)
    expTerm = (1 + w²)^(-d.κ)
    coeff = gamma(d.κ) / (sqrt(π * d.κ) * d.vth * gamma(d.κ - 0.5))
    return coeff * expTerm
end

"""
    _rand!(rng::AbstractRNG, d::Kappa, x)

Generates a random velocity vector sampled from the 3D Kappa distribution.

Algorithm:
The Kappa distribution is generated using a compound probability method (decomposition 
into a Maxwellian with a Chi-squared distributed temperature variance).

1. Sample from Chi-squared: ξ ~ ChiSq(2κ - 1)
2. Sample from Isotropic Normal: Z ~ Normal(0, I)
3. ``𝐯 = 𝐮₀ + vₜₕ * √(κ / ξ) * Z``

## References
- https://www.wikiwand.com/en/articles/Student%27s_t-distribution
- [Multivariate t-distribution](https://www.wikiwand.com/en/articles/Multivariate_t-distribution)
"""
function _rand!(rng::AbstractRNG, d::Kappa, x)
    # Derived from matching power laws: -(κ+1) == -(ν+3)/2
    ν = 2 * d.κ - 1 # degrees of freedom (ν)
    ξ = rand(rng, Chisq(ν))
    Z = randn(rng, 3)
    scale = d.vth * sqrt(d.κ / ξ) # variance scaling factor
    return x .= d.u0 .+ scale .* Z
end
