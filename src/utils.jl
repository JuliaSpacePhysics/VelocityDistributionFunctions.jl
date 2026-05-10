using Random: AbstractRNG

# like @tullio but with fastmath = false by default
macro sum(ex...)
    return esc(
        quote
            @tullio $(ex...) fastmath = false grad = false tensor = false
        end
    )
end

_finite(x::T) where {T} = ifelse(!isnan(x), x, zero(T))

# Gamma(α, 1) sampler via Marsaglia-Tsang method
function _rand_gamma(rng::AbstractRNG, α::Real)
    if α < 1
        return _rand_gamma(rng, α + 1) * rand(rng)^(1 / α)
    end
    d = α - 1 / 3
    c = 1 / sqrt(9 * d)
    while true
        x = randn(rng)
        v = (1 + c * x)^3
        if v > 0
            u = rand(rng)
            if u < 1 - 0.0331 * x^4 || log(u) < 0.5 * x^2 + d * (1 - v + log(v))
                return d * v
            end
        end
    end
end

# Chi-squared(ν) = Gamma(ν/2, 2)
_rand_chi2(rng::AbstractRNG, ν::Real) = 2 * _rand_gamma(rng, ν / 2)

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
