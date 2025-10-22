module VelocityDistributionFunctions

using Tullio: @tullio
using Bumper
using LoopVectorization
using LinearAlgebra
using StaticArrays

include("utils.jl")
include("spectra.jl")
include("pad.jl")

export pitch_angle_distribution, tpitch_angle_distribution
export directional_energy_spectra

"""
    sort_flux_by_pitch_angle(flux, pitch_angle)

Sort flux by pitch angle into ascending order. Set `sort_on_reverse` to true to sort only if the pitch angle is in reverse order.
"""
function sort_flux_by_pitch_angle!(flux, pitch_angle; sort_on_reverse = true)
    pa_dim = 1
    n_pa = size(pitch_angle, pa_dim)
    N_flux = ndims(flux)
    N_pa = ndims(pitch_angle)
    @assert size(flux, pa_dim) == n_pa
    @assert N_flux ∈ (2, 3)
    @assert N_pa == 2

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

end
