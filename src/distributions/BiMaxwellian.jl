"""
    BiMaxwellian(vth_perp, vth_para, 𝐮₀=[0, 0, 0], b0=[0, 0, 1])
    BiMaxwellian(T_perp::Temperature, T_para::Temperature, 𝐮₀=[0, 0, 0], b0=[0, 0, 1]; mass = me)

BiMaxwellian velocity distribution with different thermal velocities in perpendicular `vth_perp` and parallel `vth_para` directions and drift velocity `𝐮₀` and magnetic field direction `b0`.

```math
f(𝐯) ∝  \\exp[-\\frac{(𝐯_⟂ - 𝐮_{0, ⟂})^{2}}{v_{\\mathrm{th}, ⟂}^{2}}] \\exp[-\\frac{(𝐯_∥ - 𝐮_{0, ∥})^{2}}{v_{\\mathrm{th}, ∥}^{2}}]
```

where the normalization constant is ``A = √π^{-3} / (v_{th,∥} v_{th,⟂}^2)``.
"""
struct BiMaxwellianPDF{T, TB, TVD} <: AbstractVelocityPDF
    vth_perp::T
    vth_para::T
    b0::TB
    u0::TVD

    function BiMaxwellianPDF(
            vth_perp::T, vth_para::T = vth_perp,
            u0::TVD = _zero_𝐯(T), b0::TB = SA[0.0, 0.0, 1.0];
            check_args = true
        ) where {T, TVD, TB}
        @check_args BiMaxwellianPDF (vth_perp, vth_perp >= zero(vth_perp)) (vth_para, vth_para >= zero(vth_para)) (b0, length(b0) == 3) (u0, length(u0) == 3)
        BT = base_numeric_type(T)
        B_normalized = normalize(BT.(b0))
        return new{T, TB, TVD}(vth_perp, vth_para, B_normalized, u0)
    end
end

BiMaxwellian(args...; kw...) = BiMaxwellianPDF(args...; kw...)

function _rand!(rng::AbstractRNG, d::BiMaxwellianPDF{T}, x) where {T}
    bperp1 = normalize(d.b0 × get_least_parallel_basis_vector(d.b0))
    bperp2 = d.b0 × bperp1
    vpara = d.vth_para * randn(rng) / sqrt(2)
    vperp_1 = (d.vth_perp / sqrt(2)) * randn(rng)
    vperp_2 = (d.vth_perp / sqrt(2)) * randn(rng)
    @. x = d.u0 + vpara * d.b0 + vperp_1 * bperp1 + vperp_2 * bperp2
    return x
end

# Generalal pdf that supports unitful inputs
function _pdf(d::BiMaxwellianPDF, 𝐯::AbstractVector{T}) where {T}
    d𝐯 = 𝐯 - d.u0
    dv_para = d𝐯 ⋅ d.b0
    v_perp_sq = sum(abs2, d𝐯 - dv_para * d.b0)
    exponent = -(dv_para^2 / d.vth_para^2 + v_perp_sq / d.vth_perp^2)
    A = (π^-1.5) / (d.vth_para * d.vth_perp^2) # normalization constant A
    return A * exp(exponent)
end


"""
Normalized PDF for the parallel velocity

```math
f(v) = 1 / (√π vₜₕ) · exp[-(v/vₜₕ)²]
```
"""
function Distributions.pdf(d::BiMaxwellianPDF, v::VPar)
    return exp(-((v.val - d.u0 ⋅ d.b0) / d.vth_para)^2) / √π / d.vth_para
end

function Distributions.pdf(d::BiMaxwellianPDF, v::VPerp)
    return exp(-((v.val - d.u0 ⋅ d.b0) / d.vth_perp)^2) * 2 * v.val / d.vth_perp^2
end
