addprocs(5)

@everywhere begin
    using Distributions
    using DEMC
    srand(1)
    # set up target distribution: Multivariate Normal
    ndim = 50 # Number of dimensions
    μ = rand(ndim) # mean of each dimension
    A = rand((ndim, ndim))
    Σ = A'*A + diagm(3*ones(ndim)) # variance covariance matrix
    Σ = Σ./maximum(Σ)
    distr = MvNormal(μ, Σ)
    # log objective function
    log_obj(mean) = log(pdf(MvNormal(μ, Σ), mean))
end

# set up of DEMCz chain
Npar = length(μ)
blockindex = [1:Npar] # parameter blocks: here choose all parameters to be updated simultaenously
Nblocks = length(blockindex)
eps_scale = 1e-5*ones(Npar) # scale of random error around DE update
γ = 2.38 # scale of DE update, 2.38 is the "optimal" number for a normal distribution
N = 5
K = 10
Z = randn((10*ndim, ndim))

# Number of iterations in Chain
Nburn = 500
Ngeneration  = 100

mc_burn = DEMC.demcz_sample_par(log_obj, Z, N, K, Nburn, Nblocks, blockindex, eps_scale, γ)
chainflat = DEMC.flatten_chain(mc_burn.chain, N, Nburn, Npar)'
Z = chainflat[end-10*ndim+1:end, :] # new initial Z

mc = DEMC.demcz_sample_par(log_obj, Z, N, K, Ngeneration, Nblocks, blockindex, eps_scale, γ)


# did we converge?
convergence_check(mc.chain, mc.log_obj, N, Ngeneration, Npar, "./img/demcz_par_normal/" ; verbose = true)

# estimates
chainflat = DEMC.flatten_chain(mc.chain, N, Ngeneration, Npar)'
# bhat = mean(chainflat,1)[:]
# println("\n estimates: ", bhat, "\n dist to true: ", bhat - μ)
# bhat = median(chainflat,1)[:]
# println("\n estimates: ", bhat, "\n dist to true: ", bhat - μ)

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


h1 = histogram(chainflat[:,1], lab = "DEMCz, N = $N, T = $Ngeneration", normed=true, nbin = 33)
plot!(x1, normal1, lab="target", linewidth = 3)
h2 = histogram(chainflat[:,2], lab = "DEMCz, N = $N, T = $Ngeneration", normed=true, nbin = 33)
plot!(x2, normal2, lab="target", linewidth = 3)

p = plot(h1, h2, layout=(2,1) )
savefig(p,"./img/normal_direct_demcz_hist_1_2.png")
