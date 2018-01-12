# DE-MCMC

[![Build Status](https://travis-ci.org/chrished/DEMCMC.jl.svg?branch=master)](https://travis-ci.org/chrished/DE-MCMC.jl)

[![Coverage Status](https://coveralls.io/repos/chrished/DEMCMC.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chrished/DE-MCMC.jl?branch=master)

[![codecov.io](http://codecov.io/github/chrished/DEMCMC.jl/coverage.svg?branch=master)](http://codecov.io/github/chrished/DE-MCMC.jl?branch=master)


This repository contains
1. implementation of the "DE-MCMC" algorithm proposed by Ter Braak (2006).
2. parallelized computation of "DE-MCMC" algorithm (parallel over the different chains)
3. convergence check for MCMC method R̂ statistic as in Gelman et al. (2014).
4. convenience function to display trace of obj function value in simulation, convergence check and acceptance ratios
5. convenience function to calculate mean and covariance of simulated chains

## Dependencies
* ParallelDataTransfer.jl for (2.)
* Plots.jl and GR.jl for (4.)
* they can both be disabled if one only wants to use the serial implementation and no figures
* GR can be replaced by any other backend.

## References

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2014). Bayesian data analysis (Vol. 2). Boca Raton, FL: CRC press.

Ter Braak, C. J. (2006). A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces. Statistics and Computing, 16(3), 239-249.
