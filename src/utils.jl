# like @tullio but with fastmath = false by default
macro sum(ex...)
    return esc(
        quote
            @tullio $(ex...) fastmath = false grad = false tensor = false
        end
    )
end

_finite(x::T) where {T} = ifelse(!isnan(x), x, zero(T))