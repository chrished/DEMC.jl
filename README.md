# DEMC - Differential Evolution Markov Chain 


[![Build Status](https://travis-ci.org/chrished/DEMC.jl.svg?branch=master)](https://travis-ci.org/chrished/DEMC.jl)

[![Coverage Status](https://coveralls.io/repos/chrished/DEMC.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chrished/DEMC.jl?branch=master)

[![codecov.io](http://codecov.io/github/chrished/DEMC.jl/coverage.svg?branch=master)](http://codecov.io/github/chrished/DEMC.jl?branch=master)


This repository contains
* implementation of the "DEMC" algorithm proposed by Ter Braak (2006).
* POSSIBLY THIS IS NOT VALID: parallelized computation of "DEMC" algorithm (parallel over the different chains)
* convergence check: R̂ statistic as in Gelman et al. (2014).
* convenience function to display trace of obj function value in simulation, convergence check and acceptance ratios
* convenience function to calculate mean and covariance of simulated chains

TODO: DEMCz Algorithm and renaming of repository

## Sample Usage
For a self-contained example see `normal_direct_example.jl` in the example directory.

One example run for simulating a Multivariate Normal gives the following simulated densities for dimension 1 and 2. The figure was created by the `normal_direct_example.jl` script. Check there for details.
![Simulated Multivariate Normal](./img/normal_direct_hist_1_2.png)

Here the main setup, you only need to supply your objective function and set the right number of parameters.

```julia
using DEMC

# log_obj(x)::Function is the log likelihood which accepts a vector x of length Npar as its argument
Npar = 3 # this is the length of the parameter vector.

# setup parameters for chain
Npop = 3*Npar # this should be larger than ndim
pop_guess = randn((Npop, Npar)) # random guess to start the markov chain (this should be sufficiently dispersed to cover the relevant parameter range)
# Number of iterations in Chain
Ngeneration  = 1000
blockindex = [1:ndim] # parameter blocks: here choose all parameters to be updated simultaenously
eps_scale = 1e-4*ones(Npar) # scale of random error around DE update
γ = 2.38 # scale of DE update, 2.38 is the "optimal" number for a normal
# run chain
demc = demc_sample(log_obj, pop_guess, Ngeneration, blockindex, eps_scale, γ)
# one can also use a previous run of a chain as a starting point
demc_2ndrun = demc_sample(log_obj, demc, Ngeneration, blockindex, eps_scale, γ)
# did we converge?
Rhat = DEMC.Rhat_gelman(demc_2ndrun.chain, Npop, Ngeneration, Npar)
```


## References

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2014). Bayesian data analysis (Vol. 2). Boca Raton, FL: CRC press.

Ter Braak, Cajo JF (2006). A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces. Statistics and Computing, 16(3), 239-249.

ter Braak, Cajo JF, and Jasper A. Vrugt. "Differential evolution Markov chain with snooker updater and fewer chains." Statistics and Computing 18.4 (2008): 435-446.
