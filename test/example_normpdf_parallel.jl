# addprocs, same fun everywhere!
using Distributed
while nprocs()<4
    addprocs(1)
end
using Pkg
@everywhere using Pkg
@everywhere Pkg.activate(".")
using DEMC

using Random
using LinearAlgebra
using Distributions
using SharedArrays
using ParallelDataTransfer

@everywhere begin
    using DEMC
    using Random
    using LinearAlgebra
    using Distributions
    using SharedArrays
    Random.seed!(31953150+myid())
    using ParallelDataTransfer
end

# set up target distribution: Multivariate Normal
ndim = 10 # Number of dimensions
μ = rand(ndim) # mean of each dimension
A = rand(ndim, ndim)
Σ = A'*A .+ diagm(0 => 2*ones(ndim)) # variance covariance matrix
Σ = Σ./maximum(Σ)/100
distr = MvNormal(μ, Σ)
passobj(myid(), workers(),[:μ, :Σ, :distr])
# log objective function
@everywhere log_obj(mean) = logpdf(distr, mean)

# set up of DEMCz chain
opts = DEMC.demcopt(ndim)
opts.blockindex = [1:ndim] # parameter blocks: here choose all parameters to be updated simultaenously
opts.Nblocks = length(opts.blockindex)
opts.eps_scale = 1e-5*ones(ndim) # scale of random error around DE update
opts.γ = 2. # scale of DE update, 2.38 is the "optimal" number for a normal distribution
opts.N = nworkers() # number of chains
opts.K = 10 # every K steps add current N draws to Z
# Number of iterations in Chain
opts.Ngeneration = 100_000
opts.autostop = :Rhat
opts.autostop_Rhat = 1.075

Z = randn((10*ndim, ndim)) # initial distribution (completely off to make a difficult test case)
mc, Z = DEMC.demcz_sample_par(log_obj, Z,opts; prevrun=nothing, sync_every = 2000)

# drop first half of chain
Ntot = size(mc.chain,3)
Nburn = 10000
if Nburn >= round(Ntot/2)+1
    Nburn = Int(round(Ntot/2))
end
keep = Ntot-Nburn+1:Ntot
Ngen_burned = length(keep)
chain_burned = mc.chain[:,:,keep]
logobj_burned = mc.log_obj[:, keep]
chainflat = DEMC.flatten_chain(chain_burned, opts.N, Ngen_burned, ndim)'
# covariance of estimates
b, Σb = DEMC.mean_cov_chain(chain_burned, opts.N, Ngen_burned, ndim)
println("\n estimates: ", b, "\n dist to true: ", b - μ)

figure_path = pwd()*"/img/normpdf_parallel/"
accept_ratio, Rhat = DEMC.convergence_check(chain_burned, logobj_burned, figure_path; verbose = false)

@test all(Rhat.<1.1)
@test all(accept_ratio.> 0.1)
@test all(accept_ratio.< 0.45)

rmprocs(workers())

# sample = [rand(distr) for s=1:100_000]
# llsample = [logpdf(distr, sample[s]) for s=1:100_000]
# DEMC.histogram(logobj_burned[:], normalize=true)
# DEMC.histogram!(llsample, normalize=true)
# DEMC.savefig(figure_path*"lpdf_histogram.png")
