# ============================================================================
# Individual moment computations
# ============================================================================

# Fused weighted sums: compute flux (3), mftens (6), and eflux (3) in a single pass.
# Avoids three separate broadcasts over fE_kernel and three separate _weighted_sum calls.
@muladd function compute_fused_moments(fE_kernel, e_inf, Omega_j, Omega_ij)
    T = eltype(fE_kernel)
    n_energy = size(fE_kernel, 1)
    e1d = ndims(e_inf) == 1
    flux = zero(SVector{3, T})
    mftens = zero(SVector{6, T})
    eflux = zero(SVector{3, T})
    @inbounds for J in CartesianIndices(tail(size(fE_kernel)))
        ωj = SVector{3, T}(Omega_j[1, J], Omega_j[2, J], Omega_j[3, J])
        ωij = SVector{6, T}(
            Omega_ij[1, J], Omega_ij[2, J], Omega_ij[3, J],
            Omega_ij[4, J], Omega_ij[5, J], Omega_ij[6, J]
        )
        sf, sm, se = zero(T), zero(T), zero(T)   # accumulator for e_inf^1, e_inf^1.5, e_inf^2
        for ie in 1:n_energy
            ei = e1d ? ie : CartesianIndex(ie, Tuple(J)...)
            einf = e_inf[ei]
            tmp = fE_kernel[ie, J] * einf
            sf += tmp
            sm = tmp * sqrt(einf) + sm
            se = tmp * einf + se
        end
        flux = flux + sf * ωj
        mftens = mftens + sm * ωij
        eflux = eflux + se * ωj
    end
    return flux, mftens, eflux
end

@muladd function compute_heat_flux(tmp2, theta, phi, e_inf, mass, velocity)
    T = eltype(velocity)
    factor = T(sqrt(mass / 2) * mass / 2)
    two_over_mass = T(2 / mass)
    n_energy = size(tmp2, 1)
    e1d = ndims(e_inf) == 1
    q = zero(SVector{3, T})
    # Outer loop over angle bins: compute trig once per angle
    @inbounds for J in CartesianIndices(tail(size(tmp2)))
        sth, cth = sincosd(theta[J])
        sph, cph = sincosd(phi[J])
        ehat = SA[cth * cph, cth * sph, sth]
        # Inner loop over energy bins
        for ie in 1:n_energy
            ei = e1d ? ie : CartesianIndex(ie, Tuple(J)...)
            v = sqrt(two_over_mass * e_inf[ei])
            w = v * ehat - velocity
            c = (w ⋅ w) * tmp2[ie, J]
            q = c * w + q
        end
    end
    return q * factor
end


function compute_pressure_tensor(mftens, velocity, flux, mass)
    T = eltype(mftens)
    mf3x3 = SMatrix{3, 3, T}(
        mftens[1], mftens[4], mftens[5],
        mftens[4], mftens[2], mftens[6],
        mftens[5], mftens[6], mftens[3]
    )
    pt3x3 = mf3x3 - (velocity * flux') .* mass
    ptens = SA[pt3x3[1, 1], pt3x3[2, 2], pt3x3[3, 3], pt3x3[1, 2], pt3x3[1, 3], pt3x3[2, 3]]
    return ptens
end

"""
    compute_temperature(ptens, density, mass) -> NamedTuple

Temperature quantities from the pressure tensor.

Eigenvalue decomposition of ``\\mathbf{T}`` yields three temperature eigenvalues
``(T_1, T_2, T_3)`` sorted by the SPEDAS heuristic:
the eigenvalue whose removal makes the remaining two most similar is assigned
to slot 3.

Returns `(; ttens, avgtemp, vthermal, t3, t3evec)`.
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
    compute_field_aligned(ttens, t3evec, velocity, magf)

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

Returns `(; magt3, symm, symm_theta, symm_phi, symm_ang)`.
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
    symm_theta = asind(symm[3] / symm_r)
    symm_phi = atand(symm[2], symm[1])

    return (; magt3, symm, symm_theta, symm_phi, symm_ang)
end
