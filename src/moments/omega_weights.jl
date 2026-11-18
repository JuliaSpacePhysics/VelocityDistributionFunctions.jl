# https://github.com/spedas/pyspedas/blob/master/pyspedas/particles/moments/moments_3d_omega_weights.py

# Solid-angle integration weights of an angle bin, with centers `th` (polar), `ph` and widths `dth`, `dph` in degrees:
#
# | Index | Weight                                  | Used for              |
# |:------|:----------------------------------------|:----------------------|
# | 1     | ∫ dΩ                                    | density               |
# | 2–4   | ∫ ê dΩ, (x, y, z)                       | particle/energy flux  |
# | 5–10  | ∫ ê ê dΩ, (xx, yy, zz, xy, xz, yz)      | momentum flux tensor  |
#
# Each weight is a θ integral times a φ integral, so `_theta_factors` and `_phi_factors` evaluate the trig
# once per θ and per φ bin and `_omega` combines them per cell.
function _theta_factors(th, dth)
    th, dth = deg2rad(th), deg2rad(dth)
    st1, ct1 = sincos(th - dth / 2)
    st2, ct2 = sincos(th + dth / 2)
    it = ct1 - ct2                                   # ∫ sinθ
    it_zz = (ct1^3 - ct2^3) / 3                      # ∫ sinθ cos²θ
    return (
        it,
        dth / 2 - (st2 * ct2 - st1 * ct1) / 2,       # ∫ sin²θ
        (st2^2 - st1^2) / 2,                         # ∫ sinθ cosθ
        it - it_zz,                                  # ∫ sin³θ
        it_zz,
        (st2^3 - st1^3) / 3,                         # ∫ sin²θ cosθ
    )
end

function _phi_factors(ph, dph)
    ph, dph = deg2rad(ph), deg2rad(dph)
    sp1, cp1 = sincos(ph - dph / 2)
    sp2, cp2 = sincos(ph + dph / 2)
    h = (sp2 * cp2 - sp1 * cp1) / 2
    return (
        dph,                                         # ∫ 1
        sp2 - sp1,                                   # ∫ cosφ
        cp1 - cp2,                                   # ∫ sinφ
        dph / 2 + h,                                 # ∫ cos²φ
        dph / 2 - h,                                 # ∫ sin²φ
        (sp2^2 - sp1^2) / 2,                         # ∫ sinφ cosφ
    )
end

function _omega(t, p)
    it, it_x, it_z, it_xx, it_zz, it_xz = t
    dph, ip_x, ip_y, ip_xx, ip_yy, ip_xy = p
    return (
        it * dph, it_x * ip_x, it_x * ip_y, it_z * dph,
        it_xx * ip_xx, it_xx * ip_yy, it_zz * dph, it_xx * ip_xy, it_xz * ip_x, it_xz * ip_y,
    )
end
