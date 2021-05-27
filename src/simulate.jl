"""
    simulate(μ, x)

Simulation of a dichotomous vector via the probit link.

# Arguments
- `μ::Vector{<:Real}`: real-valued vector of coefficients
- `x::Matrix{<:Real}`: real-valued vector or matrix of inputs

# Output
- `y::Vector{Bool}`: boolean vector of outcomes
- `p::Vector{float}`: vector of probabilities
- `z::Vector{float}`: vector of latent variables
"""

function simulate(β::Vector{T} where T<:Real, α::Real, X::Matrix{T} where T<:Real)
    N, J = size(X)
    @assert length(β) == J "μ is a $(length(β)) vector and X is a $Nx$J matrix."
    @assert α > 0 "σ must be a positive scalar."
    μ = X * β
    λ = exp.(-μ .* α)
    Time = rand.(Weibull.(α, λ))
    Censor = rand.(Weibull.(α, λ))
    Indicator = Time .> Censor
    #Time = ifelse.(Indicator, Time, Censor)
    return (Time, Indicator, λ, β, α)
end