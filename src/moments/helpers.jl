# Build a CartesianIndex inserting `ie` at position `edim` among the other indices `J`.
# e.g. _fullindex(3, 2, CartesianIndex(5,7)) → CartesianIndex(5,3,7)
@inline function _fullindex(ie, edim, TJ::NTuple{N, Int}) where {N}
    return CartesianIndex(_fullindex_tuple(ie, edim, TJ))
end

@inline function _fullindex_tuple(ie, edim, TJ::NTuple{N, Int}) where {N}
    return ntuple(i -> i < edim ? TJ[i] : i == edim ? ie : TJ[i - 1], N + 1)
end


# Rotation matrix with z' along `v1`, y' along `v1 × v2`, x' completing the triad.
# Matches SPEDAS `rot_mat`.
function _rot_mat(v1, v2)
    a = normalize(v1)
    b = normalize(a × v2)
    c = b × a
    return hcat(c, b, a)   # columns = [x', y', z']
end

function _safe_eigen(A)
    T = eltype(A)
    try
        vals, vecs = eigen(A)
        return SVector{3, T}(vals), SMatrix{3, 3, T}(vecs)
    catch
        nan = T(NaN)
        return SA[nan, nan, nan], @SMatrix fill(nan, 3, 3)
    end
end

function _temp_sort(eig)
    vals, vecs = eig
    s = SVector{3, Int}(sortperm(SVector(vals)))
    num = vals[s[2]] < (vals[s[1]] + vals[s[3]]) / 2 ? s[3] : s[1]
    shft = (-1, 1, 0)[num]
    return circshift(vals, shft), vecs[:, circshift(SA[1, 2, 3], shft)]
end


_bin_difference(a, b, ::Nothing) = abs(a - b)
function _bin_difference(a, b, period)
    d = abs(a - b)
    return d > period / 2 ? period - d : d
end

function _bin_widths!(dx, x, period = nothing)
    n = length(x)
    n >= 2 || throw(ArgumentError("supply a bin width for a singleton axis"))
    @inbounds begin
        prev = _bin_difference(x[2], x[1], period)
        dx[1] = prev
        for i in 2:(n - 1)
            next = _bin_difference(x[i + 1], x[i], period)
            dx[i] = (prev + next) / 2
            prev = next
        end
        dx[n] = prev
    end
    return dx
end

_axis_widths!(dx, x::AbstractVector, dim, period) = _bin_widths!(dx, x, period)
function _axis_widths!(dx, x, dim, period)
    for J in CartesianIndices(_osize(x, dim))
        inds = _fullindex_tuple(Colon(), dim, Tuple(J))
        _bin_widths!(view(dx, inds...), view(x, inds...), period)
    end
    return dx
end

_bin_widths(x, period = nothing) = _bin_widths!(similar(x), x, period)
