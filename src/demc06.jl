"""
demc_sample(logobj, chain, ngeneration, blockindex, eps_scale)

Differential evoultion Markov Chain Metropolis Hastings in order to sample from distribution of parameters Θ whose posterior is proportional to exp(logobj(Θ)).

ngeneration: number of generations

demc_guess: starting point of markov chain, can either be a matrix of guesses with size Npopulation*Nparameter or an element of type DEMC. If it is of type DEMC, then we use the last state of the population to initialize the current chain.

blockindex: indices for sub blocks of Θ from which to sample. E.g. blockindex = [1:2, 3:5] splits 5 element vector into two blocks of the first two and the last 3 parameters.

eps_scale: rescale Normal(0,1) error term in taking steps (should be small as part of the jumps are already taking by the DE part)

γ: updating weight in differential evolution. set to -1. for standard γ = 2.38/sqrt(2*d)
"""
function demc_sample(logobj, demc_guess, Ngeneration, blockindex, eps_scale, γ)
    if typeof(demc_guess) == MC
        X = demc_guess.Xcurrent
        log_objcurrent = demc_guess.log_objcurrent
        Npop, Npar = size(X)
        demc = MC(Array{Float64}(Npop, Npar, Ngeneration),Array{Float64}(Npop, Ngeneration), X, log_objcurrent)
    else
        X = copy(demc_guess)
        Npop, Npar = size(X)
        log_objcurrent = map(logobj, [X[i,:] for i = 1:Npop])
        demc = MC(Array{Float64}(Npop, Npar, Ngeneration),Array{Float64}(Npop, Ngeneration), X, log_objcurrent)
    end

    if γ == -1.
        γ = 2.38
    end

    if blockindex == []
        Nblocks = Npar
        blockindex = collect(1:Npar)
    else
        Nblocks = length(blockindex)
    end

    # initialize HelperArrays
    helparr = HelperArrays(mkchainset(1, Npop), zeros(Npar))
    # run chain
    for in = 1:Ngeneration, ic = 1:Npop, ib = 1:Nblocks
        update_demc_chain!(in, ic, ib, demc, logobj, blockindex, eps_scale, γ, Ngeneration, Npop, Nblocks, helparr)
    end
    return demc
end


function mkchainset(i, N)
    return deleteat!(collect(1:N), i)
end

"""
update_demc_chain!(in, ic, ib, demc, obj, blockindex, eps_scale)
"""
function update_demc_chain!(in, ic, ib, demc, logobj, blockindex, eps_scale, γ, Ngeneration, Npop, Nblocks, helparr)
    current_logobj = demc.log_objcurrent[ic]
    Xproposal = demc.Xcurrent[ic,:]

    # which chains to mix with?
    helparr.chainset[:] = mkchainset(ic, Npop)
    i1 = rand(helparr.chainset)
    i2 = rand(helparr.chainset)

    block = blockindex[ib]
    # generate proposal
    helparr.de_diffvec[:] = 0.
    blocklen = length(block)
    if blocklen == 1
        helparr.de_diffvec[block] = γ*(demc.Xcurrent[i1,block]-demc.Xcurrent[i2,block]) + eps_scale[block] .* randn()
    else
        helparr.de_diffvec[block] = γ/sqrt(blocklen) *(demc.Xcurrent[i1,block]-demc.Xcurrent[i2,block]) + eps_scale[block] .* randn(blocklen)
    end

    Xproposal = Xproposal + helparr.de_diffvec
    # calculate likelihood
    log_objXprop = logobj(Xproposal)
    #println("prop: ", log_objX)
    #println("current: ", demc.log_objcurrent[ic])
    # acceptance step
    if log(rand()) <  log_objXprop - demc.log_objcurrent[ic]
        demc.chain[ic, :, in] = Xproposal
        demc.log_obj[ic, in] = log_objXprop
        demc.Xcurrent[ic, :] = Xproposal
        demc.log_objcurrent[ic] = log_objXprop
        #demc.accept[ic] += 1
    else
        demc.chain[ic, :, in] = demc.Xcurrent[ic,:]
        demc.log_obj[ic, in] = demc.log_objcurrent[ic]
    end
end

"""
update_demc_chain(in, ic, ib, current_logobj, Xproposal, logobj, blockindex, eps_scale, γ, Ngeneration, Npop, Nblocks, helparr)
"""

function update_demc_chain(in, ic, ib, current_logobj, Xcurrent, logobj, blockindex, eps_scale, γ, Ngeneration, Npop, Nblocks)
    Xproposal = Xcurrent[ic,:]

    # which chains to mix with?
    chainset = mkchainset(ic, Npop)
    i1 = rand(chainset)
    i2 = rand(chainset)

    block = blockindex[ib]
    # generate proposal
    de_diffvec = zeros(Xproposal)
    blocklen = length(block)
    if blocklen == 1
        de_diffvec[block] = γ*(Xcurrent[i1,block]-Xcurrent[i2,block]) + eps_scale[block] .* randn()
    else
        de_diffvec[block] = γ/sqrt(blocklen)*(Xcurrent[i1,block]-Xcurrent[i2,block]) + eps_scale[block] .* randn(blocklen)
    end

    Xproposal = Xproposal + de_diffvec
    # calculate likelihood
    log_objXprop = logobj(Xproposal)

    #println("prop: ", log_objX)
    #println("current: ", demc.log_objcurrent[ic])
    # acceptance step

    if log(rand()) < log_objXprop - current_logobj[ic]
        return(Xproposal, log_objXprop)
    else
        return(Xcurrent[ic,:], current_logobj[ic])
    end
end

"""
Rewrite DEMC parallel sampling
same as  demc_sample, but calculates each subchain in parallel at each iteration across generations.
"""
function demc_sample_par(logobj, demc_guess, Ngeneration, blockindex, eps_scale, γ)
    wp = CachingPool(workers())
    # so each chain picks different random numbers
    @everywhere srand(myid())
    # initialize arrays
    if typeof(demc_guess) == MC
        global Xcurrent = copy(demc_guess.Xcurrent)
        global log_objcurrent = copy(demc_guess.log_objcurrent)
        Npop, Npar = size(Xcurrent)
        demc = MC(Array{Float64}(Npop, Npar, Ngeneration),Array{Float64}(Npop, Ngeneration),Array(Xcurrent), Array(log_objcurrent), Array(zeros(Int,Npop)))
    else
        global Xcurrent = copy(demc_guess)
        Npop, Npar = size(Xcurrent)
        global log_objcurrent = pmap(wp, logobj, [Xcurrent[i,:] for i = 1:Npop])
        demc = MC(Array{Float64}(Npop, Npar, Ngeneration),Array{Float64}(Npop, Ngeneration),Array(Xcurrent), Array(log_objcurrent), Array(zeros(Int,Npop)))
    end

    @everywhere global Xcurrent
    @everywhere global log_objcurrent
    passobj(myid(), workers(), [:Xcurrent, :log_objcurrent], from_mod = DEMCMC, to_mod = DEMCMC)

    if γ < 0.
        γ = 2.38
    end

    if blockindex == []
        Nblocks = Npar
        blockindex = collect(1:Npar)
    else
        Nblocks = length(blockindex)
    end



    # initialize array and results vector
    in = 1
    res = pmap(wp,ic -> runblocks(in, ic,  Xcurrent, log_objcurrent, logobj, blockindex, eps_scale, γ, Ngeneration, Npop, Nblocks), 1:Npop)

    for ic = 1:Npop
        demc.chain[ic,:,in] = res[ic][1]
        demc.log_obj[ic,in] = res[ic][2]
        demc.Xcurrent[ic, :] = res[ic][1]
        demc.log_objcurrent[ic] = res[ic][2]
        Xcurrent[ic, :] = res[ic][1]
        log_objcurrent[ic] = res[ic][2]
    end
    passobj(myid(), workers(), [:Xcurrent, :log_objcurrent], from_mod = DEMCMC, to_mod = DEMCMC)

    for in = 2:Ngeneration
        # pmap over ic
        res[:] = pmap(wp,ic -> DEMCMC.runblocks(in, ic,  Xcurrent, log_objcurrent, logobj, blockindex, eps_scale, γ, Ngeneration, Npop, Nblocks), 1:Npop)
        for ic = 1:Npop
            demc.chain[ic,:,in] = res[ic][1]
            demc.log_obj[ic,in] = res[ic][2]
            demc.Xcurrent[ic, :] = res[ic][1]
            demc.log_objcurrent[ic] = res[ic][2]
            Xcurrent[ic, :] = res[ic][1]
            log_objcurrent[ic] = res[ic][2]
        end
        passobj(myid(), workers(), [:Xcurrent, :log_objcurrent], from_mod = DEMCMC, to_mod = DEMCMC)
    end

    return demc
end


function runblocks(in, ic, current_pop, current_logobj, logobj, blockindex, eps_scale, γ, Ngeneration, Npop, Nblocks)
    for ib = 1:Nblocks
        newX, newobj = update_demc_chain(in, ic, ib, current_logobj, current_pop, logobj, blockindex, eps_scale, γ, Ngeneration, Npop, Nblocks)
        current_pop[ic,:] = newX
        current_logobj = newobj
    end
    return current_pop[ic,:], current_logobj
end
