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
    for i in 2:draws
        # Propose new values
        b[i, :], a[i] = proposal(b[i-1, :], a[i-1], β, α, Ts, X)
        
        # Time - update missing values (i.e. censored values)
        for n in 1:N
            if Indicator[n] == 1
                λ = exp(-dot(X[n, :], b[i, :]) * a[i])
                Ts[n] = rand(Truncated(Weibull(a[i], λ), Time[n], Inf))
            end
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

function proposal(b, a, β, α, Time, X)
    # Existing log-likelihood
    pra = sum(logpdf(α, a))     # Log-prior for alpha
    prb = sum(logpdf(β, b))     # Log-prior for beta
    ll  = loglik(b, a, Time, X) # Log-likelihood
    po  = pra + prb + ll        # Log-posterior

    # New log-likelihood
    a1 = exp(rand(Normal(log(a), 0.05))) # Proposal for alpha
    b1 = rand.(Normal.(b, 0.05))         # Proposal for beta

    pra = sum(logpdf(α, a1))      # Log-prior for alpha
    prb = sum(logpdf(β, b1))      # Log-prior for beta
    ll  = loglik(b1, a1, Time, X) # Log-likelihood
    po1 = pra + prb + ll          # Log-posterior
    
    ratio = exp(po1 - po)         # Log-posterior
    if rand() < ratio
        return (b1, a1)
    else
        return (b, a)
    end
end

N = 1_000;
X = [ones(N) repeat(0:1, inner = convert(Int, N/2))];
Time, Indicator, λ, β, α = simulate([-0.5, 1.0], 1.0, X);
model = estimate(Time, Indicator, X, MvNormal([0, 0], I), Exponential(1), 5_000)

mean(Time[X[:, 2] .== 0])
mean(Time[X[:, 2] .== 1])

#X = ones(N, 1);
#Time, Indicator = simulate([0.0], 1.0, X);
#model = estimate(Time, Indicator, X, MvNormal([0], 10_000*I), Exponential(1))
using StatsPlots
#histogram(model[1])
plot(model[1], color = [:blue :red])
hline!(β', color = [:blue :red], label = false)

#Fit = [mean(Weibull(model[2][i], exp(-dot(X[n, :], model[1][i, :]) * model[2][i]))) for n in 1:N, i in 501:1500];
#Fit = mean(Fit, dims = 2);

#plot(Time)
#plot!(Fit)