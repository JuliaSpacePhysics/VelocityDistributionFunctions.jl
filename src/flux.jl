"""
    sort_flux_by_pitch_angle!(flux, pitch_angle; sort_on_reverse = true)

Sort flux by pitch angle into ascending order. Set `sort_on_reverse` to true to sort only if the pitch angle is in reverse order.
"""
function sort_flux_by_pitch_angle!(flux, pitch_angle; sort_on_reverse = true)
    pa_dim = 1
    n_pa = size(pitch_angle, pa_dim)
    N_flux = ndims(flux)
    N_pa = ndims(pitch_angle)
    size(flux, pa_dim) == n_pa || throw(DimensionMismatch("flux has $(size(flux, pa_dim)) pitch angles along dim 1, pitch_angle has $n_pa"))
    N_flux ∈ (2, 3) || throw(ArgumentError("flux must be 2- or 3-dimensional, got $N_flux dimensions (size $(size(flux)))"))
    N_pa == 2 || throw(ArgumentError("pitch_angle must be 2-dimensional [pitch, time], got $N_pa dimensions (size $(size(pitch_angle)))"))

    # Sort each time step
    perm = zeros(Int, n_pa)
    T = promote_type(eltype(flux), eltype(pitch_angle))
    cache = zeros(T, n_pa)
    @inbounds for (flux_slice, pa_slice) in zip(eachslice(flux, dims = N_flux), eachslice(pitch_angle, dims = N_pa))
        sortperm!(perm, pa_slice)
        sort_on_reverse && perm != n_pa:-1:1 && continue
        _sort_by_perm!(pa_slice, cache, perm)
        for f in eachcol(flux_slice)
            _sort_by_perm!(f, cache, perm)
        end
    end
    return flux, pitch_angle
end

function _sort_by_perm!(x, cache, perm)
    for k in eachindex(x, cache)
        cache[k] = x[perm[k]]
    end
    return x .= cache
end