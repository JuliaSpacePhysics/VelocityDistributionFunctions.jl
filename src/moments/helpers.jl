# Column-major strides of an array of padded size `shape` (see `_bshape`), zero along its singleton axes,
# so data cell `t` broadcasts from its linear index `_offset(t, strides)`.
_bstrides(shape) = map((n, st) -> n == 1 ? 0 : st, shape, Base.size_to_strides(1, shape...))
@inline _offset(t, strides) = 1 + sum(map((i, st) -> (i - 1) * st, t, strides); init = 0)

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
    if !all(isfinite, A)
        nan = T(NaN)
        return SA[nan, nan, nan], @SMatrix fill(nan, 3, 3)
    end
    vals, vecs = eigen(A)
    return SVector{3, T}(vals), SMatrix{3, 3, T}(vecs)
end

# SPEDAS order: slot 3 gets the eigenvalue whose removal leaves the other two most similar.
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

function _bin_widths!(dx, x, period)
    n = length(x)
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
    all(>(0), dx) || throw(ArgumentError("adjacent bin centers coincide; supply bin widths"))
    return dx
end

function _bin_widths(x::AbstractArray, dims; period = nothing)
    dx = similar(x, float(eltype(x)))
    other = filter(!=(dims), ntuple(identity, ndims(x)))
    foreach((d, c) -> _bin_widths!(d, c, period), eachslice(dx; dims = other), eachslice(x; dims = other))
    return dx
end
