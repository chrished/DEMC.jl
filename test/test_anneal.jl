using DEMC
using Random
using LinearAlgebra
using Distributions
Random.seed!(31953150)
# set up target distribution: Multivariate Normal
ndim = 10 # Number of dimensions
μ = rand(ndim) # mean of each dimension
# log objective function
log_obj(mean) = -sum((mean.-μ).^2)

# set up of DEMCz chain
Npar = length(μ)
blockindex = [1:Npar] # parameter blocks: here choose all parameters to be updated simultaenously
Nblocks = length(blockindex)
eps_scale = 1e-5*ones(Npar) # scale of random error around DE update
γ = 2.38 # scale of DE update, 2.38 is the "optimal" number for a normal distribution
N = 5 # number of chains
K = 10 # every K steps add current N draws to Z
Z = randn((10*ndim, ndim)) # initial distribution (completely off to make a difficult test case)

# Number of iterations in Chain
Ngen = 5000
mc, Z = DEMC.demcz_anneal(log_obj, Z, N, K, Ngen, Nblocks, blockindex, eps_scale, γ; verbose=false, TN = 1e-4, T0 = 2)

# check best element
bestval = maximum(mc.log_obj)
bestel = findfirst(mc.log_obj.==bestval)
bestpar = mc.chain[bestel[1], :, bestel[2]]

@test abs(bestval)>-1e-1

#bestvals = [maximum(mc.log_obj[:, ig]) for ig = 1:Ngen]
#DEMC.plot(1:Ngen, bestvals)
