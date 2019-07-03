using Pkg; Pkg.activate(pwd())
using DEMC
using Random
using LinearAlgebra
using Distributions
Random.seed!(319531501)

# generate data
nobs = 1000
npar = 25
Σ = ones(npar, npar)
for i = 1:npar
    for iprime = 1:i
        if i != iprime
            Σ[i, iprime] = 0.25
            Σ[iprime, i] = 0.25
        end
    end
end

DataDistribution = MvNormal(zeros(npar), Σ)
X = zeros(nobs, npar+1)
X[:,1] .= 1.
for iobs = 1:nobs
    X[iobs, 2:end] .= rand(DataDistribution)
end

β = 1 .+ rand(npar+1)*3
y = X*β + randn(nobs)

# log objective function - for correct std errors need to use optimal weights
log_obj(b) = -0.5 * sum((y.-X*b).^2)


# set up of DEMCz chain
ndim = length(β)
opts = DEMC.demcopt(ndim)
opts.blockindex = [1:ndim] # parameter blocks: here choose all parameters to be updated simultaenously
opts.Nblocks = length(opts.blockindex)
opts.eps_scale = 1e-5*ones(ndim) # scale of random error around DE update
opts.γ = 2. # scale of DE update, 2.38 is the "optimal" number for a normal distribution
opts.N = 5 # number of chains
opts.K = 10 # every K steps add current N draws to Z
# Number of iterations in Chain
opts.Ngeneration = 100000
opts.autostop = :Rhat
opts.autostop_every = 2000
opts.autostop_Rhat = 1.1
opts.print_step = 1000
opts.verbose = false

Z = randn((10*ndim, ndim))

# run chain
mc, Z = DEMC.demcz_sample(log_obj, Z, opts)

# drop first half of chain
Ntot = size(mc.chain,3)
keep = Int(Ntot-opts.autostop_every)+1:Ntot
Ngen_burned = length(keep)
chain_burned = mc.chain[:,:,keep]
chainflat = DEMC.flatten_chain(chain_burned, N, Ngen_burned, Npar)'

# covariance of estimates
b, Σb = DEMC.mean_cov_chain(chain_burned)
# correct std errors for incorrect weighting matrix
# OLS
bols = (X'*X)\(X'*y)
ui = y .- X*bols
s2hat = (1/(nobs-npar)) * sum(ui.^2)
Σols = inv(X'X)*s2hat

# difference simulated to OLS
println("\n MCMC estimates: ", b, "\n dist to OLS: ", b .- bols)

accept_ratio, Rhat = DEMC.convergence_check(mc.chain, mc.log_obj, "none"; verbose = false, parnames = [])
