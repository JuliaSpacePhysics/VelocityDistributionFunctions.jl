# Build a CartesianIndex inserting `ie` at position `edim` among the other indices `J`.
# e.g. _fullindex(3, 2, CartesianIndex(5,7)) → CartesianIndex(5,3,7)
@inline function _fullindex(ie, edim, TJ::NTuple{N, Int}) where {N}
    tuple = ntuple(i -> i < edim ? TJ[i] : i == edim ? ie : TJ[i - 1], N + 1)
    return CartesianIndex(tuple)
end

_scalar(x, i) = x isa Number ? x : x[i]
_vector(::Nothing, i) = nothing
_vector(x::AbstractVector, i) = x
_vector(x::AbstractMatrix, i) = selectdim(x, 1, i)

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


# Estimate energy bin widths from bin centers.
# de[i] = energy[i] - energy[i-1]  (backward difference)
# denergy[i] = (de[i+1] + de[i]) / 2  (shifted average)
# Edges: denergy[1] = de[2], denergy[end] = de[end]
function _compute_denergy!(dE, energy)
    n = size(energy, 1)
    @inbounds for J in CartesianIndices(axes(energy)[2:end])
        # backward differences: de[i] = energy[i] - energy[i-1]
        # (de[1] is undefined, start from 2)
        # interior: dE[i] = (de[i+1] + de[i]) / 2 for i in 2:n-1
        for i in 2:(n - 1)
            de_i = energy[i, J] - energy[i - 1, J]
            de_ip1 = energy[i + 1, J] - energy[i, J]
            dE[i, J] = (de_ip1 + de_i) / 2
        end
        dE[1, J] = energy[2, J] - energy[1, J]
        dE[n, J] = energy[n, J] - energy[n - 1, J]
    end
    return dE
end

# Compute bin widths from bin centers using centered finite differences.
# When `period` is given (e.g. 360 for azimuthal angles), differences that
# exceed half the period are treated as circular wraps.
function _bin_widths!(dx, x, period = nothing)
    @assert length(x) > 1
    n = length(x)

    _cdiff(a, b) = (d = abs(a - b); !isnothing(period) && d > period / 2 ? period - d : d)

    dx[1] = _cdiff(x[2], x[1])
    @inbounds for i in 2:(n - 1)
        dx[i] = (_cdiff(x[i + 1], x[i]) + _cdiff(x[i], x[i - 1])) / 2
    end
    dx[n] = _cdiff(x[n], x[n - 1])
    return dx
end

_bin_widths(x, period = nothing) = _bin_widths!(similar(x), x, period)
