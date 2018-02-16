using Distributions
using DEMC

# set up target distribution: Multivariate Normal
ndim = 3 # Number of dimensions
μ = rand(ndim) # mean of each dimension
A = rand((ndim, ndim))
Σ = A'*A + diagm(3*ones(ndim)) # variance covariance matrix
distr = MvNormal(μ, Σ)
# log objective function
log_obj(mean) = log(pdf(MvNormal(μ, Σ), mean))

# make guess for parameter population
Npop = 3*ndim
Npar = length(μ)
pop_guess = randn((Npop, Npar))
# Number of iterations in Chain
Nburn = 3000
Ngeneration  = 5000

# set up of DEMC chain
blockindex = [1:ndim] # parameter blocks: here choose all parameters to be updated simultaenously
eps_scale = 1e-4*ones(Npar) # scale of random error around DE update
γ = 2.38 # scale of DE update, 2.38 is the "optimal" number for a normal distribution

# Now run markov chain for initial burn period
demc_burn = demc_sample(log_obj, pop_guess, Nburn, blockindex, eps_scale, γ)
# Now run markov chain taking as starting point the demc_burn chain
demc = demc_sample(log_obj, demc_burn, Ngeneration, blockindex, eps_scale, γ)

# did we converge?
Rhat = DEMC.Rhat_gelman(demc.chain, Npop, Ngeneration, Npar)
println("Gelmans Rhat: $Rhat")

# estimates
chainflat = DEMC.flatten_chain(demc.chain, Npop, Ngeneration, Npar)'
bhat = mean(chainflat,1)[:]
println("\n estimates: ", bhat, "\n dist to true: ", bhat - μ)

# covariance of estimates
cov_bhat = cov(chainflat)


# plot simulated vs true distribution
using Plots
gr()
using Distributions


se = sqrt(diag(Σ))
x1 = linspace(μ[1]-4*se[1], μ[1] + 4*se[1], 200)
normal1 = pdf(Normal(μ[1], se[1]),x1)
x2 =  linspace(μ[2]-4*se[2], μ[2] + 4*se[2], 200)
normal2 = pdf(Normal(μ[2], se[2]),x2)


h1 = histogram(chainflat[:,1], lab = "DE MCMC", normed=true, nbin = 33)
plot!(x1, normal1, lab="target", linewidth = 3)
h2 = histogram(chainflat[:,2], lab = "DE MCMC", normed=true, nbin = 33)
plot!(x2, normal2, lab="target", linewidth = 3)

p = plot(h1, h2, layout=(2,1) )
savefig(p,"./img/normal_direct_hist_1_2.png")
