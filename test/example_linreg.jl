using DEMC
using Random
using LinearAlgebra
using Distributions
Random.seed!(31953150)

# generate data
nobs = 100
npar = 3
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

β = rand(npar+1)
y = X*β + randn(nobs)

# log objective function - for correct std errors need to use optimal weights
log_obj(b) = -0.5 * sum((y.-X*b).^2)

# set up of DEMCz chain
Npar = length(β)
ndim = Npar
blockindex = [1:Npar] # parameter blocks: here choose all parameters to be updated simultaenously
Nblocks = length(blockindex)
eps_scale = 1e-5*ones(Npar) # scale of random error around DE update
γ = 2.38 # scale of DE update, 2.38 is the "optimal" number for a normal distribution
N = 3
K = 10
Z = randn((10*ndim, ndim))

# Number of iterations in Chain
Ngen = 2500
mc, Z = DEMC.demcz_sample(log_obj, Z, N, K, Ngen, Nblocks, blockindex, eps_scale, γ; verbose =false)

# drop first half of chain
Ntot = size(mc.chain,3)
keep = Int(Ntot-round(Ngen/2))+1:Ntot
Ngen_burned = length(keep)
chain_burned = mc.chain[:,:,keep]
chainflat = DEMC.flatten_chain(chain_burned, N, Ngen_burned, Npar)'
bhat = mean(chainflat,dims=1)[:]
println("\n estimates: ", bhat, "\n dist to true: ", bhat - β)
# covariance of estimates
b, Σb = DEMC.mean_cov_chain(chain_burned, N, Ngen_burned, Npar)
# correct std errors for incorrect weighting matrix

# OLS
bols = (X'*X)\(X'*y)
ui = y .- X*bols
s2hat = (1/(nobs-npar)) * sum(ui.^2)
Σols = inv(X'X)*s2hat
