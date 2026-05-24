# Interface similar to pyspedas to facilitate cross-validation

"""
    tmoments(dists::AbstractVector, sc_pots; kw...)
    tmoments(dists::AbstractVector, sc_pots, magfs; kw...)

Batch-compute plasma moments for a vector of pre-processed distributions,
returning a `StructArray`.
"""
function tmoments(dists::AbstractVector, sc_pots; kw...)
    structT = Base.promote_op(plasma_moments, eltype(dists), eltype(sc_pots))
    result = StructArray{structT}(undef, length(dists))
    Threads.@threads :dynamic for i in eachindex(dists, sc_pots)
        result[i] = plasma_moments(dists[i], sc_pots[i]; kw...)
    end
    return result
end

function tmoments(dists::AbstractVector, sc_pots, magfs; kw...)
    structT = Base.promote_op(plasma_moments, eltype(dists), eltype(sc_pots), eltype(magfs))
    result = StructArray{structT}(undef, length(dists))
    Threads.@threads :dynamic for i in eachindex(dists, sc_pots, magfs)
        result[i] = plasma_moments(dists[i], sc_pots[i], magfs[i]; kw...)
    end
    return result
end
