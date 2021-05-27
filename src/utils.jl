"""
    ESS(chain, k)

Calculate the Effective Sample Size.

# Arguments
- `chain`: matrix of draws from the target distribution
- `k`: (integer) number of lags

# Output
- `ESS`: vector of effective sample sizes
"""

function ESS(chain::Matrix{T} where T<:AbstractFloat, k = 10::Integer)
    M, J = size(chain)
    @assert M > k "The chain of β must be longer than k."
    ρ = sum(pacf(chain, 1:k); dims = 1)
    τ = 1 .+ 2 .* sum(ρ; dims = 1)
    ESS = round.(Int, vec(M ./ τ))
    return ESS
end

"""
    probability(x, chain)

Calculate probabilities based on the probit link.ß

# Arguments
- `x`: matrix of inputs
- `chain`: matrix of draws from the target distribution

# Output
- `p`: matrix of probabilities
"""

function probability(x::Matrix{T} where T<:Real, chain::Matrix{T} where T<:AbstractFloat)
    z = x * chain'
    p = cdf(Normal(0, 1), z)
    return p
end
probability(x::Vector{T} where T<:Real, chain::Matrix{T} where T<:AbstractFloat) = probability(repeat(x, 1, 1), chain)