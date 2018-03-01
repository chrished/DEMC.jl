function demcz_anneal(logobj, Zmat, N, K, Ngeneration, Nblocks, blockindex, eps_scale, γ, tempfun)
    M, d = size(Zmat)
    X = Zmat[end-N+1:end, :]
    log_objcurrent = map(logobj, [X[i,:] for i = 1:N])
    mc = MC(Array{Float64}(N,  d, Ngeneration),Array{Float64}(N, Ngeneration), X, log_objcurrent)
    temp = 1.

    bestval = maximum(log_objcurrent)
    bestpar = X[findfirst(bestval.==log_objcurrent), :]
    println("iteration 0")
    println("bestval = $bestval")
    println("bestpar = $bestpar")
    for ig = 1:Ngeneration
        temp = tempfun(ig)
        for ic = 1:N
            Xcurrent, current_logobj = update_blocks_anneal(mc.Xcurrent[ic, :], mc.log_objcurrent[ic], Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks,temp)
            # update in chain
            mc.chain[ic, :, ig] = Xcurrent
            mc.log_obj[ic, ig] = current_logobj
            mc.Xcurrent[ic, :] = Xcurrent
            mc.log_objcurrent[ic] = current_logobj
        end
        bestval = maximum(mc.log_objcurrent)
        bestpar[:] = mc.Xcurrent[findfirst(bestval.==mc.log_objcurrent), :]
        if mod(ig, K) == 0.
            Zmat = vcat(Zmat, mc.Xcurrent)
            M += N
            println("iteration $ig")
            println("bestval = $bestval")
            println("bestpar = $bestpar")
        end
    end
    return mc
end

function demcz_anneal_par(logobj, Zmat, N, K, Ngeneration, Nblocks, blockindex, eps_scale, γ, tempfun)
    wp = CachingPool(workers())
    Mval, d = size(Zmat)
    X = Zmat[end-N+1:end, :]
    log_objcurrent = pmap(wp, logobj, [X[i,:] for i = 1:N])
    mc = MC(Array{Float64}(N,  d, Ngeneration),Array{Float64}(N, Ngeneration), X, log_objcurrent)
    mc.Xcurrent = X
    mc.log_objcurrent = log_objcurrent
    global Xcurrent = copy(mc.Xcurrent)
    global log_objcurrent = copy(mc.log_objcurrent)
    global Z = Zmat
    global M = Mval
    global temp = 1.
    @everywhere global temp
    @everywhere global Z
    @everywhere global M
    @everywhere global Xcurrent
    @everywhere global log_objcurrent

    passobj(myid(), workers(), [:Xcurrent, :log_objcurrent, :temp], from_mod = DEMC, to_mod = DEMC)
    passobj(myid(), workers(), [:Z, :M], from_mod = DEMC, to_mod = DEMC)
    bestval = maximum(log_objcurrent)
    bestpar = Xcurrent[findfirst(bestval.==log_objcurrent), :]
    println("iteration 0")
    println("bestval = $bestval")
    println("bestpar = $bestpar")

    for ig = 1:Ngeneration
        temp = tempfun(ig)
        passobj(myid(), workers(), [:Xcurrent, :log_objcurrent, :temp], from_mod = DEMC, to_mod = DEMC)
        res = pmap(wp, ic -> update_blocks_anneal(Xcurrent[ic,:], log_objcurrent[ic], Z, M, logobj, blockindex, eps_scale, γ, Nblocks, temp), 1:N)
        for ic = 1:N
            # update in chain
            mc.chain[ic, :, ig] = res[ic][1]
            mc.log_obj[ic, ig] = res[ic][2]
            mc.Xcurrent[ic, :] = res[ic][1]
            mc.log_objcurrent[ic] = res[ic][2]
            Xcurrent[ic, :] = res[ic][1]
            log_objcurrent[ic] = res[ic][2]
        end
        bestval = maximum(log_objcurrent)
        bestpar[:] = Xcurrent[findfirst(bestval.==log_objcurrent), :]
        if mod(ig, K) == 0.
            Z = vcat(Z, mc.Xcurrent)
            M += N
            passobj(myid(), workers(), [:Z, :M], from_mod = DEMC, to_mod = DEMC)
            println("iteration $ig")
            println("bestval = $bestval")
            println("bestpar = $bestpar")
        end
    end
    return mc
end


function update_blocks_anneal(Xcurrent, current_logobj, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks, temp)
    for ib in 1:Nblocks
        Xcurrent[:], current_logobj = update_demcz_chain_block_anneal(Xcurrent, current_logobj, ib, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks, temp)
    end
    return Xcurrent, current_logobj
end


function update_demcz_chain_block_anneal(Xcurrent, current_logobj, ib, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks, temp)
    Xproposal = copy(Xcurrent)
    # generate proposal
    set = collect(1:M)
    i1 = rand(set)
    deleteat!(set, i1)
    i2 = rand(set)
    de_diffvec = zeros(Xproposal)
    block = blockindex[ib]
    blocklen = length(block)
    if blocklen == 1
        de_diffvec[block] = γ*(Zmat[i1,block]- Zmat[i2,block]) + eps_scale[block] .* randn()
    else
        de_diffvec[block] = γ/sqrt(2*blocklen)*(Zmat[i1,block]-Zmat[i2,block]) + eps_scale[block] .* randn(blocklen)
    end
    Xproposal += de_diffvec
    log_objXprop = logobj(Xproposal)
    if temp == 0.
        temp += 1e-9
    end
    if  log(rand()*temp +1.-temp) < (log_objXprop - current_logobj)
        return Xproposal, log_objXprop
    else
        return Xcurrent, current_logobj
    end
end
