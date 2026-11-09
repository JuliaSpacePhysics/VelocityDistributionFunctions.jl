# https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d_omega_weights.py

_theta(theta, J) = theta[J]
_theta(theta::AbstractVector, J::CartesianIndex{2}) = theta[J[2]]
_theta(theta::Real, J) = theta
_phi(phi, J) = phi[J]
_phi(phi::AbstractVector, J::CartesianIndex{2}) = phi[J[1]]
_phi(phi::Real, J) = phi

"""
    omega_weights(theta, phi, dtheta, dphi)

Compute solid-angle integration weights for moments calculation.

Angles are in radians.

The 10 weight channels are:

| Index | Weight                 | Used for            |
|:------|:-----------------------|:--------------------|
| 1     | ``\\int d\\Omega``       | density             |
| 2–4   | ``\\hat{x},\\hat{y},\\hat{z}`` flux   | particle/energy flux |
| 5–10  | ``xx,yy,zz,xy,xz,yz`` | momentum flux tensor|

Bin center `theta` is the polar angle.
"""
function omega_weights(th, ph, dth, dph)
    st1, ct1 = sincos(th - dth / 2)
    st2, ct2 = sincos(th + dth / 2)
    sp1, cp1 = sincos(ph - dph / 2)
    sp2, cp2 = sincos(ph + dph / 2)

    # θ integrals of the direction-cosine products, including the sinθ of dΩ
    it = ct1 - ct2                                   # ∫ sinθ
    it_x = dth / 2 - (st2 * ct2 - st1 * ct1) / 2     # ∫ sin²θ
    it_z = (st2^2 - st1^2) / 2                       # ∫ sinθ cosθ
    it_xx = it - (ct1^3 - ct2^3) / 3                 # ∫ sin³θ
    it_zz = (ct1^3 - ct2^3) / 3                      # ∫ sinθ cos²θ
    it_xz = (st2^3 - st1^3) / 3                      # ∫ sin²θ cosθ
    # φ integrals
    ip_x = sp2 - sp1                                 # ∫ cosφ
    ip_y = cp1 - cp2                                 # ∫ sinφ
    ip_xx = dph / 2 + (sp2 * cp2 - sp1 * cp1) / 2    # ∫ cos²φ
    ip_yy = dph / 2 - (sp2 * cp2 - sp1 * cp1) / 2    # ∫ sin²φ
    ip_xy = (sp2^2 - sp1^2) / 2                      # ∫ sinφ cosφ
    return (
        it * dph, it_x * ip_x, it_x * ip_y, it_z * dph,
        it_xx * ip_xx, it_xx * ip_yy, it_zz * dph, it_xx * ip_xy, it_xz * ip_x, it_xz * ip_y,
    )
end

# Bin edges in degrees
omega_weightsd(th, ph, dth, dph) =
    omega_weights(deg2rad(th), deg2rad(ph), deg2rad(dth), deg2rad(dph))

function omega_weights!(omega, theta, phi, dtheta, dphi)
    dims = tail(size(omega))
    @inbounds for I in CartesianIndices(dims)
        weights = omega_weightsd(_theta(theta, I), _phi(phi, I), _theta(dtheta, I), _phi(dphi, I))
        omega[1:10, I] .= weights
    end
    return omega
end
