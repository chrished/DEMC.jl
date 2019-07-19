# addprocs, same fun everywhere!
using Distributed
while nprocs()<4
    addprocs(1)
end
@everywhere using Pkg
@everywhere Pkg.activate(".")
using DEMC

@everywhere using ParallelDataTransfer
@everywhere using DEMC
#@everywhere include("./src/DEMC.jl")
@everywhere using Random
@everywhere Random.seed!(31953150 + myid())
# set up target distribution: Multivariate Normal
ndim = 30 # Number of dimensions
μ = rand(ndim) # mean of each dimension
passobj(myid(), workers(), :μ)
# log objective function
@everywhere log_obj(mean) = -sum((mean.-μ).^2)

# set up of DEMCz chain
Npar = length(μ)
# set up of DEMCz chain
opts = DEMC.demcopt(Npar)
opts.blockindex = [1:Npar] # parameter blocks: here choose all parameters to be updated simultaenously
opts.Nblocks = length(opts.blockindex)
opts.eps_scale = 1e-5*ones(Npar) # scale of random error around DE update
opts.γ = 2.38 # scale of DE update, 2.38 is the "optimal" number for a normal distribution
opts.N = nworkers() # number of chains
opts.K = 10 # every K steps add current N draws to Z
# Number of iterations in Chain
opts.Ngeneration = 20_000
opts.TN = 0.0
opts.T0 = 5.0

Z = randn((10*Npar, Npar))*10 # initial distribution (completely off to make a difficult test case)

# Number of iterations in Chain
mc, Z = DEMC.demcz_anneal_par(log_obj, Z, opts; sync_every=500)

# check best element
bestval, bestel = findmax(mc.log_obj)
bestpar = mc.chain[bestel[1], :, bestel[2]]

@test abs(bestval)>-1e-1

#bestvals = [maximum(mc.log_obj[:, ig]) for ig = 1:Ngen]
#DEMC.plot(1:Ngen, bestvals)
