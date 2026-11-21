const _Series = AbstractVector{<:AbstractArray}

# Number of scans; the grid and each named input must be shared or one entry per scan, and `vectors` one three-component vector per scan.
function _scan_count(scans, grid; vectors...)
    Base.require_one_based_indexing(scans)
    nt = length(scans)
    nt > 0 || throw(ArgumentError("need at least one scan"))
    grid isa EnergyAngleBins || _check_series(grid, nt, "grid")
    for (name, v) in pairs(vectors)
        isnothing(v) && continue
        _check_series(v, nt, name)
        all(x -> (x isa AbstractVector || x isa Tuple) && length(x) == 3, v) ||
            throw(DimensionMismatch("$name needs one three-component vector per scan"))
    end
    return nt
end

function _check_series(x, nt, name)
    x isa AbstractVector && axes(x, 1) == Base.OneTo(nt) ||
        throw(DimensionMismatch("$name needs one entry for each of $nt scans"))
end

_at(x, i) = x
_at(x::AbstractVector, i) = x[i]

# `[f(1), …, f(nt)]`, threaded; `f(1)` fixes the element type.
function _map_scans(f, nt)
    first_result = f(1)
    out = Vector{typeof(first_result)}(undef, nt)
    out[1] = first_result
    Threads.@threads :dynamic for i in 2:nt
        out[i] = f(i)
    end
    return out
end
