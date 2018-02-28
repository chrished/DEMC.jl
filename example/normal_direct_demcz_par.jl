# Try parallel demcz
addprocs(3)
@everywhere begin
    using Distributions
    using DEMC
    using ParallelDataTransfer
    srand(1)
    # set up target distribution: Multivariate Normal
    ndim = 12 # Number of dimensions
    μ = rand(ndim) # mean of each dimension
    A = rand((ndim, ndim))
    Σ = A'*A + diagm(3*ones(ndim)) # variance covariance matrix
    Σ = Σ./maximum(Σ)
    distr = MvNormal(μ, Σ)
    # log objective function
    log_obj(mean) = log(pdf(MvNormal(μ, Σ), mean))
end
@everywhere srand(myid())
@everywhere println("μ: $μ")
@everywhere println("obj at true mean: ", log_obj(μ))

# set up of DEMCz chain
Npar = length(μ)
blockindex = [1:Npar] # parameter blocks: here choose all parameters to be updated simultaenously
Nblocks = length(blockindex)
eps_scale = 1e-5*ones(Npar) # scale of random error around DE update
γ = 2.38 # scale of DE update, 2.38 is the "optimal" number for a normal distribution
N = 3
K = 10
Z = randn((10*ndim, ndim))*5

# Number of iterations in Chain
Nburn = 10000
Ngeneration = 10000

mc_burn = demcz_sample_par(log_obj, Z, N, K, Nburn, Nblocks, blockindex, eps_scale, γ)
convergence_check(mc_burn.chain, mc_burn.log_obj, N, Nburn, Npar, "./img/demcz_par_burn/" ; verbose = true)

chainflat = DEMC.flatten_chain(mc_burn.chain, N, Nburn, Npar)'
Z = chainflat[end-10*ndim+1:end, :] # new initial Z
# run final chain
mc = DEMC.demcz_sample_par(log_obj, Z, N, K, Ngeneration, Nblocks, blockindex, eps_scale, γ)

# did we converge?
convergence_check(mc.chain, mc.log_obj, N, Ngeneration, Npar, "./img/demcz_par_normal/" ; verbose = true)

# estimates
chainflat = DEMC.flatten_chain(mc.chain, N, Ngeneration, Npar)'
# covariance of estimates
b, Σb = mean_cov_chain(mc.chain, N, Ngeneration, Npar)


#
# # plot simulated vs true distribution
# using Plots
# gr()
# using Distributions
#
#
# se = sqrt(diag(Σ))
# x1 = linspace(μ[1]-4*se[1], μ[1] + 4*se[1], 200)
# normal1 = pdf(Normal(μ[1], se[1]),x1)
# x2 =  linspace(μ[2]-4*se[2], μ[2] + 4*se[2], 200)
# normal2 = pdf(Normal(μ[2], se[2]),x2)
#
#
# h1 = histogram(chainflat[:,1], lab = "DEMCz, N = $N, T = $Ngeneration", normed=true, nbin = 33)
# plot!(x1, normal1, lab="target", linewidth = 3)
# h2 = histogram(chainflat[:,2], lab = "DEMCz, N = $N, T = $Ngeneration", normed=true, nbin = 33)
# plot!(x2, normal2, lab="target", linewidth = 3)
#
# p = plot(h1, h2, layout=(2,1) )
# savefig(p,"./img/normal_direct_demcz_hist_1_2.png")
