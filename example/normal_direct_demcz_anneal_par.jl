@everywhere begin
    using Distributions
    using DEMC
    srand(1)
    # set up target distribution: Multivariate Normal
    ndim = 11 # Number of dimensions
    μ = rand(ndim) # mean of each dimension
    A = rand((ndim, ndim))
    Σ = A'*A + diagm(3*ones(ndim)) # variance covariance matrix
    Σ = Σ./maximum(Σ)
    distr = MvNormal(μ, Σ)
    # log objective function
    log_obj(mean) = log(pdf(MvNormal(μ, Σ), mean))
end

@everywhere srand(myid())
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
Ngeneration  = 6000
tfun(x) = max(0.,1. - (x/Ngeneration))
mc = DEMC.demcz_anneal_par(log_obj, Z, N, K, Ngeneration, Nblocks, blockindex, eps_scale, γ, tfun)

maxf, maxpar = extract_best(mc, N)

println("best value: $maxf")
println("max value: ", log_obj(μ))
