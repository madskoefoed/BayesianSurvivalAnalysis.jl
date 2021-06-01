"""
    Metropolis(y, x, β₀, β, M)

A Metropolis algorithm for probit regression.

# Arguments
- `y::Vector{Bool}`: boolean vector of outcomes
- `x::Matrix{<:Real}`: real valued vector or matrix of inputs
- `β₀::MvNormal`: a univariate or multivariate normal distribution for the prior
- `β::MvNormal`: a univariate or multivariate normal distribution for the candidate
- `M::Integer`: the number of draws

# Output
- `chain::Matrix{<:Float}`: matrix of draws from the target distribution
- `accept::Vector{Bool}`: boolean vector of acceptance indicators

The function can be called without 'x' in which case a constant-only model is estimated.
"""

function estimate(Time::Vector{T} where T<:Real,
                  Indicator::BitVector,
                  X::Matrix{T} where T<:Real,
                  β::MvNormal,
                  α::Exponential,
                  draws = 10_000::Integer)
    
    N, J = size(X)
    @assert length(Time) == N "Time and X must have the same number of rows."
    @assert length(Indicator) == N "Indicator and X must have the same number of rows."
    @assert J == length(β.μ) "The number of columns in x must match the dimension of β."

    # Initial values
    b = zeros(draws, J)
    a = zeros(draws)

    b[1, :] = β.μ #rand(β)
    a[1] = α.θ #rand(α)

    Ts = copy(Time)
    Ss = findall(Indicator .== 1)
    for i in 2:draws
        # Propose new values
        b[i, :], a[i] = proposal(b[i-1, :], a[i-1], β, α, Ts, X)

        # Time - update missing values (i.e. censored values)
        #λ = lambda(b[i, :], a[i], X[Ss, :])
        #Ts[Ss] = rand.(Truncated.(Weibull.(a[i], λ), Time[Ss], Inf))
        for n in Ss
            λ = exp(-dot(X[n, :], b[i, :]) * a[i])
            Ts[n] = rand(Truncated(Weibull(a[i], λ), Time[n], Inf))
        end
    end
    return (b, a)
end

function lambda(b::Vector, a::AbstractFloat, X::Matrix)
    m = X * b
    λ = exp.(-m * a)
    return λ
end

function loglik(b::Vector, a::AbstractFloat, Time::Vector, X::Matrix)
    λ = lambda(b, a, X)
    ll = sum(logpdf.(Weibull.(a, λ), Time))
    return ll
end

function logprior(prior, x)
    return sum(logpdf(prior, x))
end

function proposal(b, a, β, α, Time, X)
    # Existing
    pr = logprior(α, a) + logprior(β, b) # Log-prior
    ll = loglik(b, a, Time, X)           # Log-likelihood
    po = pr + ll                         # Log-posterior

    # Proposal
    a1 = exp(rand(Normal(log(a), 0.05))) # Proposal for alpha
    b1 = rand.(Normal.(b, 0.1))         # Proposal for beta

    pr1 = logprior(α, a1) + logprior(β, b1) # Log-prior
    ll1 = loglik(b1, a1, Time, X)           # Log-likelihood
    po1 = pr1 + ll1                         # Log-posterior

    if log(rand()) < (po1 - po)
        return (b1, a1)
    else
        return (b, a)
    end
end