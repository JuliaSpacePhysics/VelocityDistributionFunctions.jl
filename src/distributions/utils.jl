"""
    @check_args(DistName, (value, predicate), ...)

If the local `check_args::Bool` is true, evaluate each predicate and throw
`DomainError(value, "DistName: the condition <predicate> is not satisfied.")` on failure.
"""
macro check_args(D, conds...)
    checks = map(conds) do cond
        Meta.isexpr(cond, :tuple) || error("@check_args: each condition must be a tuple (value, predicate)")
        value, pred = cond.args
        msg = string(D, ": the condition ", pred, " is not satisfied.")
        :($(esc(pred)) || throw(DomainError($(esc(value)), $msg)))
    end
    return quote
        if $(esc(:check_args))
            $(checks...)
        end
    end
end


# Marsaglia & Tsang (2000) gamma sampler. Requires shape α ≥ 1.
function _rand_gamma(rng::AbstractRNG, α)
    α ≥ 1 || throw(DomainError(α, "_rand_gamma requires α ≥ 1"))
    d = α - 1 / 3
    c = 1 / sqrt(9 * d)
    while true
        Z = randn(rng)
        Z > -1 / c || continue
        v = (1 + c * Z)^3
        U = rand(rng)
        if log(U) < Z^2 / 2 + d * (1 - v + log(v))
            return d * v
        end
    end
    return
end

"""
    _rand_chisq(rng, ν)

Sample from the chi-squared distribution with `ν` degrees of freedom. Implemented
via `Gamma(ν/2, 2)` using Marsaglia & Tsang; requires `ν ≥ 2`.
"""
_rand_chisq(rng::AbstractRNG, ν) = 2 * _rand_gamma(rng, ν / 2)
