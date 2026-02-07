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
