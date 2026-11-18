# Broadcast indexing: data cell `I` reads cell `_project(I, shape)` of an array of padded size `shape` (see `_bshape`).
@inline _project(I, shape) = CartesianIndex(map((i, s) -> s == 1 ? 1 : i, Tuple(I), shape))

# `(J, members)` for each cell `J` of an array of padded size `shape`: the data cells that broadcast onto it.
# Loops only over the axes the array lacks, which is faster than a padded `CartesianIndices`.
function _cells(shape, data)
    dims = Tuple(d for d in 1:ndims(data) if shape[d] == 1 && size(data, d) > 1)
    group = CartesianIndices(map(d -> axes(data, d), dims))
    return ((J, (_member(J, K, dims) for K in group)) for J in CartesianIndices(shape))
end

# `J` with its entries at `dims` replaced by those of `K`
@inline _member(J, K, dims) = CartesianIndex(_setindices(Tuple(J), Tuple(K), dims))
@inline _setindices(t, ::Tuple{}, ::Tuple{}) = t
@inline _setindices(t, k, d) = _setindices(Base.setindex(t, first(k), first(d)), tail(k), tail(d))

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
