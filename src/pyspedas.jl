# Interface similar to pyspedas to facilitate cross-validation

"""
    tmoments(dists::AbstractVector, sc_pots; kw...)
    tmoments(dists::AbstractVector, sc_pots, magfs; kw...)

Batch-compute plasma moments for a vector of pre-processed distributions,
returning a `StructArray`.
"""
function tmoments(dists::AbstractVector, sc_pots; kw...)
    structT = Base.return_types(plasma_moments, Tuple{eltype(dists), eltype(sc_pots), Nothing})[1]
    result = StructArray{structT}(undef, length(dists))
    tforeach(eachindex(dists, sc_pots)) do i
        result[i] = plasma_moments(dists[i], sc_pots[i]; kw...)
    end
    return result
end

function tmoments(dists::AbstractVector, sc_pots, magfs; kw...)
    structT = Base.return_types(plasma_moments, Tuple{eltype(dists), eltype(sc_pots), eltype(magfs)})[1]
    result = StructArray{structT}(undef, length(dists))
    tforeach(eachindex(dists, sc_pots, magfs)) do i
        result[i] = plasma_moments(dists[i], sc_pots[i], magfs[i]; kw...)
    end
    return result
end
