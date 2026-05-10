"""
A Kappa distribution has a nearly Maxwellian core at low energies, and highenergy tails decreasing as suprathermal power laws that can be significantly broader than exponential tails.

See also [pierrardSuprathermalPopulationsTheir2021](@citet) and [pierrardKappaDistributionsTheory2010](@citet).
"""
abstract type KappaDistribution{T, K} <: AbstractVelocityPDF{T} end

"""
    KappaPDF(vth, κ)
    KappaPDF(T::Temperature, κ; mass = me)
    
Kappa velocity distribution with index `κ` and thermal velocity `vth`.

```math
f(𝐯) ∝ [1 + |𝐯|²/(κ·vₜₕ²)]^{-(κ+1)}
```

where the normalization constant is ``A_3 = Γ(κ + 1) / Γ(κ - 1/2) / (π κ v_{th}^2)^{3/2}``.

# Notes
Kappa index must be > 1.5 for finite variance. For large κ, the distribution approaches a Maxwellian. Smaller κ values produce stronger high-energy tails.

See also [`kappa_thermal_speed`](@ref).
"""
struct KappaPDF{T, K <: Real} <: KappaDistribution{T, K}
    vth::T
    κ::K

    function KappaPDF(vth::T, κ::K; check_args = true) where {T, K}
        @check_args KappaPDF (κ, κ > 1.5) (vth, vth > zero(vth))
        return new{T, K}(vth, κ)
    end
end

"""
    kappa_thermal_speed(vth, κ)
    kappa_thermal_speed(T::Temperature, κ, m)  # requires Unitful

Return the most probable speed of a (modified) kappa distribution.

The second form (accepting `Temperature`) is available when `Unitful` is loaded.

```math
V_{th,κ} = v_{th} \\sqrt{\\frac{κ - 3/2}{κ}}
```
"""
kappa_thermal_speed(vth, κ) = vth * sqrt((κ - 3/2) / κ)


_Aκ(κ, vth) = gamma(κ + 1) / gamma(κ - 1 / 2) / √((π * κ)^3) / vth^3

_pdf(d::KappaPDF, 𝐯) = _pdf_v²(d, sum(abs2, 𝐯))

function _pdf_v²(d::KappaPDF, v²)
    w² = v² / (d.κ * d.vth^2)
    expTerm = (1 + w²)^(-d.κ - 1)
    return _Aκ(d.κ, d.vth) * expTerm
end

function _pdf_1d(d::KappaPDF, vx)
    w² = vx^2 / (d.κ * d.vth^2)
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
3. ``𝐯 = vₜₕ * √(κ / ξ) * Z``

## References
- https://www.wikiwand.com/en/articles/Student%27s_t-distribution
- [Multivariate t-distribution](https://www.wikiwand.com/en/articles/Multivariate_t-distribution)
"""
function _rand!(rng::AbstractRNG, d::KappaPDF, x)
    # Derived from matching power laws: -(κ+1) == -(ν+3)/2
    ν = 2 * d.κ - 1 # degrees of freedom (ν)
    ξ = rand(rng, Chisq(ν))
    Z = randn(rng, 3)
    scale = d.vth * sqrt(d.κ / ξ) # variance scaling factor
    return x .= scale .* Z
end
