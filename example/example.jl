# Example with 3 independent variables (constant, dummy, and continuous variable)
using StatsPlots

N = 1000;
X = [ones(N) rand(0:1, N) randn(N)];
Time, Indicator, λ, β, α = simulate([0.5, 1.0, -1.0], .5, X);
b, a = estimate(Time, Indicator, X, MvNormal([0, 0, 0], 100I), Exponential(1), 2_500);

# Plotting
gr(size = (800, 600))
plot(layout = (3, 1), label = false)

plot!(b, color = [:blue :red :green], subplot = 1, label = false)
hline!(β', color = [:blue :red :green], label = false, subplot = 1)

plot!(a, color = :grey, subplot = 2, label = false)
hline!([α], color = :black, label = false, linewidth = 2, subplot = 2)

Fit = lambda(mean(b[501:2500, :]; dims = 1)[:], mean(a[501:2500]), X)
#scatter!(Time, Fit, subplot = 3, label = false)
#plot!(0:10, 0:10, subplot = 3, label = false)
plot!([Time Fit], subplot = 3, label = false)