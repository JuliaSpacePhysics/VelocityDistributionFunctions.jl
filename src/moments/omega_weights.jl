# https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d_omega_weights.py

_theta(theta, J) = theta[J]
_theta(theta::AbstractVector, J::CartesianIndex{2}) = theta[J[2]]
_theta(theta::Real, J) = theta
_phi(phi, J) = phi[J]
_phi(phi::AbstractVector, J::CartesianIndex{2}) = phi[J[1]]
_phi(phi::Real, J) = phi

# Solid angle per bin: dΩ ∝ |sin(θ+dθ/2) - sin(θ-dθ/2)| · dφ
function _domega(θ, dθ, dφ)
    sth2 = sin(θ + dθ / 2)
    sth1 = sin(θ - dθ / 2)
    return abs(sth2 - sth1) * dφ
end

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

Follows the IDL/PySPEDAS `moments_3d_omega_weights` convention (latitude angles).
"""
function omega_weights(th, ph, dth, dph)
    sth1, cth1 = sincos(th - dth / 2)
    sth2, cth2 = sincos(th + dth / 2)
    sph1, cph1 = sincos(ph - dph / 2)
    sph2, cph2 = sincos(ph + dph / 2)

    ict = sth2 - sth1
    icp = sph2 - sph1
    isp = -cph2 + cph1

    is2p = dph / 2 - sph2 * cph2 / 2 + sph1 * cph1 / 2
    ic2p = dph / 2 + sph2 * cph2 / 2 - sph1 * cph1 / 2
    ic2t = dth / 2 + sth2 * cth2 / 2 - sth1 * cth1 / 2
    ic3t = sth2 - sth1 - (sth2^3 - sth1^3) / 3
    ictst = (sth2^2 - sth1^2) / 2
    icts2t = (sth2^3 - sth1^3) / 3
    ic2tst = (-cth2^3 + cth1^3) / 3
    icpsp = (sph2^2 - sph1^2) / 2
    # solid angle
    # flux x, y, z
    # momentum flux tensor xx, yy, zz, xy, xz, yz
    return (ict * dph, ic2t * icp, ic2t * isp, ictst * dph, ic3t * ic2p, ic3t * is2p, icts2t * dph, ic3t * icpsp, ic2tst * icp, ic2tst * isp)
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
