module BayesianSurvivalAnalysis

using Distributions: Weibull, Normal, MvNormal, pdf, cdf, logpdf, Truncated, Exponential
using LinearAlgebra: I, dot
using StatsBase: sample, mean
using SpecialFunctions: gamma

# Load files
#include("./src/utils.jl")
include("./src/simulate.jl")
include("./src/estimate.jl")

# Export
#export gibbs,
#metropolis,
#simulate,
#probability,
#latent,
#effective_sample_size

end
