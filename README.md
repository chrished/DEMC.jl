# DEMC - Differential Evolution Markov Chain Monte Carlo


[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chrished.github.io/DEMC.jl/dev)
[![Build Status](https://travis-ci.com/chrished/DEMC.jl.svg?branch=master)](https://travis-ci.com/chrished/DEMC.jl)
[![Codecov](https://codecov.io/gh/chrished/DEMC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chrished/DEMC.jl)
[![Coveralls](https://coveralls.io/repos/github/chrished/DEMC.jl/badge.svg?branch=master)](https://coveralls.io/github/chrished/DEMC.jl?branch=master)


* implementation of the "DEMCz"  algorithm proposed in Ter Braak and Vrugt (2008)


## Use Cases
* useful for simulating distributions that are **not easily differentiable**, have moderate dimensionality (>1, <5?), and dimensions are potentially **highly correlated**. One example is to simulate parameter distributions in indirect inference estimators.

* moderate dimensionality is somewhat unclear, tests work alright with 10-20 dimensions, but behavior deteriorates. See https://mc-stan.org/users/documentation/case-studies/curse-dims.html


## Sample Usage
```julia
using Distributions
using DEMC

μ = zeros(5)
A = rand(5,5)
Σ = A'*A
distr = MvNormal(μ, Σ)
logobj(mean) = logpdf(distr, mean)
Zinit = rand(distr, 100)'
# sample from distr using standard options
opts = DEMC.demcopt(ndim)
mc, Z = DEMC.demcz_sample(logobj, Zinit, opts)
# see tests for further examples (also annealing and parallel)

# options you can set
# fieldnames(typeof(opts))
# :N - number of chains
# :K - add current draw to Z every K steps
# :Ngeneration - total number of steps
# :Nblocks - number of blocks
# :blockindex - subset of parameters in each block
# :eps_scale - scale of random draw around DE step
# :γ - scale of DE step (2.38 for normal distribution)
# :verbose - print avg value and avg parameters of chain  
# :print_step - print every ... steps
# :T0 - initial temperature (only for annealing)
# :TN - final temperature (only for annealing)
```
## References

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2014). Bayesian data analysis (Vol. 2). Boca Raton, FL: CRC press.

Ter Braak, Cajo JF (2006). A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces. Statistics and Computing, 16(3), 239-249.

ter Braak, Cajo JF, and Jasper A. Vrugt. "Differential evolution Markov chain with snooker updater and fewer chains." Statistics and Computing 18.4 (2008): 435-446.
