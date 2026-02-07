# https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d_omega_weights.py

"""
    omega_weights!(omega, theta, phi, dtheta, dphi)

Compute solid-angle integration weights for moments calculation.

`omega` should be an array of shape `(10, size(theta)...)` to store the
results. The 10 weight channels are:

| Index | Weight                 | Used for            |
|:------|:-----------------------|:--------------------|
| 1     | ``\\int d\\Omega``       | density             |
| 2–4   | ``\\hat{x},\\hat{y},\\hat{z}`` flux   | particle/energy flux |
| 5–10  | ``xx,yy,zz,xy,xz,yz`` | momentum flux tensor|

Follows the IDL/PySPEDAS `moments_3d_omega_weights` convention (latitude angles).
"""
function omega_weights!(omega, theta, phi, dtheta, dphi)
    T = promote_type(eltype(theta), eltype(phi), eltype(dtheta), eltype(dphi), Float64)
    dims = size(theta)
    @inbounds for I in CartesianIndices(dims)
        th, ph = T(theta[I]), T(phi[I])
        dth, dph = T(dtheta[I]), T(dphi[I])

        # Bin edges in radians
        th1, th2 = deg2rad(th - dth / 2), deg2rad(th + dth / 2)
        ph1, ph2 = deg2rad(ph - dph / 2), deg2rad(ph + dph / 2)

        sth1, cth1 = sincos(th1)
        sth2, cth2 = sincos(th2)
        sph1, cph1 = sincos(ph1)
        sph2, cph2 = sincos(ph2)

        ip = deg2rad(dph)
        ict = sth2 - sth1
        icp = sph2 - sph1
        isp = -cph2 + cph1

        is2p = deg2rad(dph / 2) - sph2 * cph2 / 2 + sph1 * cph1 / 2
        ic2p = deg2rad(dph / 2) + sph2 * cph2 / 2 - sph1 * cph1 / 2
        ic2t = deg2rad(dth / 2) + sth2 * cth2 / 2 - sth1 * cth1 / 2
        ic3t = sth2 - sth1 - (sth2^3 - sth1^3) / 3
        ictst = (sth2^2 - sth1^2) / 2
        icts2t = (sth2^3 - sth1^3) / 3
        ic2tst = (-cth2^3 + cth1^3) / 3
        icpsp = (sph2^2 - sph1^2) / 2

        omega[1, I] = ict * ip       # solid angle
        omega[2, I] = ic2t * icp     # flux x
        omega[3, I] = ic2t * isp     # flux y
        omega[4, I] = ictst * ip     # flux z
        omega[5, I] = ic3t * ic2p    # mf xx
        omega[6, I] = ic3t * is2p    # mf yy
        omega[7, I] = icts2t * ip    # mf zz
        omega[8, I] = ic3t * icpsp   # mf xy
        omega[9, I] = ic2tst * icp   # mf xz
        omega[10, I] = ic2tst * isp   # mf yz
    end
    return omega
end

function omega_weights(theta, phi, dtheta, dphi)
    omega = similar(theta, 10, size(theta)...)
    return omega_weights!(omega, theta, phi, dtheta, dphi)
end