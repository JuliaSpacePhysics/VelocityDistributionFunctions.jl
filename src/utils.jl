# like @tullio but with fastmath = false by default
macro sum(ex...)
    return esc(
        quote
            @tullio $(ex...) fastmath = false grad = false tensor = false
        end
    )
end

_finite(x::T) where {T} = ifelse(!isnan(x), x, zero(T))

# Define a temporary vector e_perp
# This selects the standard basis vector (e1, e2, or e3) that is LEAST parallel to B_dir.
# The least parallel vector gives the largest cross product magnitude (best numerical stability).
function get_least_parallel_basis_vector(𝐫)
    ST = SVector{3, eltype(𝐫)}
    return if abs(𝐫[1]) < abs(𝐫[2]) && abs(𝐫[1]) < abs(𝐫[3])
        ST(1.0, 0.0, 0.0)
    elseif abs(𝐫[2]) < abs(𝐫[3])
        ST(0.0, 1.0, 0.0)
    else
        ST(0.0, 0.0, 1.0)
    end
end
