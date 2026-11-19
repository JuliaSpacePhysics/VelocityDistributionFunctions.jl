# ============================================================================
# Individual moment computations
# ============================================================================

# Density, flux, mftens and eflux sums Σₐ S[a, p] Omega[a] over angle cells a, the linear indices of `Omega`.
function compute_fused_moments(data, W, Omega, S)
    _energy_sums!(S, data, W, size(Omega))
    T = eltype(S)
    den = zero(T)
    flux = zero(SVector{3, T})
    mftens = zero(SVector{6, T})
    eflux = zero(SVector{3, T})
    @inbounds for (a, ω) in enumerate(Omega)
        ωj = SVector{3, T}(ω[2], ω[3], ω[4])
        ωij = SVector{6, T}(ω[5], ω[6], ω[7], ω[8], ω[9], ω[10])
        den = muladd(S[a, 1], ω[1], den)
        flux = muladd.(S[a, 2], ωj, flux)
        mftens = muladd.(S[a, 3], ωij, mftens)
        eflux = muladd.(S[a, 4], ωj, eflux)
    end
    sums = (den, flux, mftens, eflux)
    any(isfinite, data) && return sums
    return map(x -> x * T(NaN), sums)
end

# S[a, p] = Σ f[I] W[e(I)][p] over the data cells I of angle cell a, where `ashape` and `size(W)` are padded
# shapes (see `_bshape`). Data are read in memory order: along axis 1 the angle cell either advances (scatter)
# or stays (reduce), and both loops vectorize.
function _energy_sums!(S, data, W, ashape)
    fill!(S, zero(eltype(S)))
    astr, wstr = _bstrides(ashape), _bstrides(size(W))
    a1, w1 = first(astr), first(wstr)
    @inbounds for J in CartesianIndices(tail(axes(data)))
        t = Tuple(J)
        a0, e0 = _offset(t, tail(astr)), _offset(t, tail(wstr))
        if iszero(a1)
            s = _reduce_energy(data, W, t, e0, w1)
            for p in 1:4
                S[a0, p] += s[p]
            end
        else
            @simd for i in axes(data, 1)
                f = _finite(data[i, t...])
                w = W[e0 + (i - 1) * w1]
                a = a0 + i - 1
                S[a, 1] = muladd(f, w[1], S[a, 1])
                S[a, 2] = muladd(f, w[2], S[a, 2])
                S[a, 3] = muladd(f, w[3], S[a, 3])
                S[a, 4] = muladd(f, w[4], S[a, 4])
            end
        end
    end
    return S
end

@inline function _reduce_energy(data, W, t, e0, w1)
    T = eltype(eltype(W))
    s1 = s2 = s3 = s4 = zero(T)
    @inbounds @simd for i in axes(data, 1)
        f = _finite(data[i, t...])
        w = W[e0 + (i - 1) * w1]
        s1 = muladd(f, w[1], s1)
        s2 = muladd(f, w[2], s2)
        s3 = muladd(f, w[3], s3)
        s4 = muladd(f, w[4], s4)
    end
    return (s1, s2, s3, s4)
end

_finite(d) = ifelse(isfinite(d), d, zero(d))

# Heat flux q = (m/2)∫|v - u|²(v - u) f d³v expanded in the lab-frame moments
function compute_heat_flux(eflux, mftens, velocity, density, mass)
    M = _symmetric(mftens)
    u = velocity
    return eflux - M * u - tr(M) / 2 * u + mass * density * (u ⋅ u) * u
end

# 3×3 matrix of a symmetric tensor stored as (xx, yy, zz, xy, xz, yz)
_symmetric(t) = SMatrix{3, 3}(t[1], t[4], t[5], t[4], t[2], t[6], t[5], t[6], t[3])

function compute_pressure_tensor(mftens, velocity, flux, mass)
    pt3x3 = _symmetric(mftens) - (velocity * flux') .* mass
    ptens = SA[pt3x3[1, 1], pt3x3[2, 2], pt3x3[3, 3], pt3x3[1, 2], pt3x3[1, 3], pt3x3[2, 3]]
    return ptens
end

"""
    compute_temperature(ptens, density, mass) -> (; ttens, avgtemp, vthermal, t3, t3evec)

Temperature quantities from the pressure tensor.

Eigenvalue decomposition of ``\\mathbf{T}`` yields three temperature eigenvalues
``(T_1, T_2, T_3)`` sorted by the SPEDAS heuristic:
the eigenvalue whose removal makes the remaining two most similar is assigned
to slot 3.
"""
function compute_temperature(ptens::SVector{6, T}, density, mass) where {T}
    pt3x3 = SMatrix{3, 3, T}(
        ptens[1], ptens[4], ptens[5],
        ptens[4], ptens[2], ptens[6],
        ptens[5], ptens[6], ptens[3]
    )
    t3x3 = pt3x3 / density
    avgtemp = (t3x3[1, 1] + t3x3[2, 2] + t3x3[3, 3]) / 3
    vth = sqrt(T(2) * avgtemp / mass)

    # Eigenvalue decomposition
    t3_eig = _safe_eigen(Symmetric(t3x3))
    t3, t3evec = _temp_sort(t3_eig)
    return (; ttens = t3x3, avgtemp, vthermal = vth, t3 = t3, t3evec = t3evec)
end

"""
    compute_field_aligned(ttens, t3evec, velocity, magf) -> (; magt3, symm, symm_theta, symm_phi, symm_ang)

Field-aligned temperature decomposition and symmetry analysis.

Rotates the temperature tensor into a field-aligned coordinate system where
``\\hat{z}' \\parallel \\mathbf{B}``:

```math
\\mathbf{T}' = R^{-1}\\, \\mathbf{T}\\, R
```

The rotation matrix ``R`` has columns ``(\\hat{x}', \\hat{y}', \\hat{z}')`` where
``\\hat{z}' = \\hat{B}``, ``\\hat{y}' = \\hat{B} \\times \\mathbf{V} / |\\hat{B} \\times \\mathbf{V}|``,
and ``\\hat{x}'`` completes the right-handed triad.

The symmetry axis is the eigenvector closest to ``\\mathbf{B}``
(with sign chosen so the dot product is positive), and the symmetry angle is:

```math
\\alpha = \\arccos|\\hat{B} \\cdot \\hat{e}_\\parallel|
```
"""
function compute_field_aligned(ttens, t3evec, velocity, magf)
    T = eltype(ttens)
    magf_sv = SVector{3, T}(magf)
    rot = _rot_mat(magf_sv, velocity)
    magt3x3 = inv(rot) * ttens * rot
    magt3 = SA[magt3x3[1, 1], magt3x3[2, 2], magt3x3[3, 3]]

    magfn = normalize(magf_sv)
    symm_col = SVector{3, T}(t3evec[:, 3])
    dot_val = magfn ⋅ symm_col
    symm_ang = acosd(clamp(abs(dot_val), zero(T), one(T)))
    symm = dot_val < 0 ? -symm_col : symm_col

    symm_r = norm(symm)
    symm_theta = acosd(symm[3] / symm_r)
    symm_phi = atand(symm[2], symm[1])

    return (; magt3, symm, symm_theta, symm_phi, symm_ang)
end
