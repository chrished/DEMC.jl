using Distributed
addprocs(3)
#using DEMC
@everywhere begin
    include("./src/DEMC.jl")
    using Random
    using LinearAlgebra
    using Distributions
    using SharedArrays
    Random.seed!(31953150+myid())
    using ParallelDataTransfer
end
# set up target distribution: Multivariate Normal
ndim = 5 # Number of dimensions
μ = rand(ndim) # mean of each dimension
A = rand(ndim, ndim)
Σ = A'*A .+ diagm(0 => 2*ones(ndim)) # variance covariance matrix
Σ = Σ./maximum(Σ)/100
distr = MvNormal(μ, Σ)
passobj(myid(), workers(),[:μ, :Σ, :distr])
# log objective function
@everywhere log_obj(mean) = logpdf(distr, mean)

# set up of DEMCz chain
Npar = length(μ)
blockindex = [1:Npar] # parameter blocks: here choose all parameters to be updated simultaenously
Nblocks = length(blockindex)
eps_scale = 1e-5*ones(Npar) # scale of random error around DE update
γ = 2.38 # scale of DE update, 2.38 is the "optimal" number for a normal distribution
N = nworkers() # number of chains
K = 10 # every K steps add current N draws to Z
Z = randn((10*ndim, ndim)) # initial distribution (completely off to make a difficult test case)

# Number of iterations in Chain
Ngen = 10000

Z = randn((10*ndim, ndim)) # initial distribution (completely off to make a difficult test case)
mc, Z = DEMC.demcz_sample_par(log_obj, Z, N, K, Ngen, Nblocks, blockindex, eps_scale, γ; prevrun=nothing)

# drop first half of chain
Ntot = size(mc.chain,3)
keep = Int(Ntot-round(Ngen/2))+1:Ntot
Ngen_burned = length(keep)
chain_burned = mc.chain[:,:,keep]
logobj_burned = mc.log_obj[:, keep]
chainflat = DEMC.flatten_chain(chain_burned, N, Ngen_burned, Npar)'
bhat = mean(chainflat,dims=1)[:]
println("\n estimates: ", bhat, "\n dist to true: ", bhat - μ)
# covariance of estimates
b, Σb = DEMC.mean_cov_chain(chain_burned, N, Ngen_burned, Npar)

figure_path = "./img/normpdf_parallel/"
accept_ratio, Rhat = DEMC.convergence_check(chain_burned, logobj_burned, figure_path; verbose = false)

@test all(Rhat.<1.1)
@test all(accept_ratio.> 0.1)
@test all(accept_ratio.< 0.45)

rmprocs(workers())
