# DEMCMC - Differential Evolution Markov Chain Monte Carlo simulation


[![Build Status](https://travis-ci.org/chrished/DEMCMC.jl.svg?branch=master)](https://travis-ci.org/chrished/DE-MCMC.jl)

[![Coverage Status](https://coveralls.io/repos/chrished/DEMCMC.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chrished/DE-MCMC.jl?branch=master)

[![codecov.io](http://codecov.io/github/chrished/DEMCMC.jl/coverage.svg?branch=master)](http://codecov.io/github/chrished/DE-MCMC.jl?branch=master)


This repository contains
* implementation of the "DE-MCMC" algorithm proposed by Ter Braak (2006).
* parallelized computation of "DE-MCMC" algorithm (parallel over the different chains)
* convergence check: R̂ statistic as in Gelman et al. (2014).
* convenience function to display trace of obj function value in simulation, convergence check and acceptance ratios
* convenience function to calculate mean and covariance of simulated chains


## Sample Usage
to be updated
## References

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2014). Bayesian data analysis (Vol. 2). Boca Raton, FL: CRC press.

Ter Braak, C. J. (2006). A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces. Statistics and Computing, 16(3), 239-249.
