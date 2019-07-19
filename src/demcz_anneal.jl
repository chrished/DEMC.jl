function tempbaseline(ig, Ng, T0, TN)
     return T0*(TN/T0)^(ig/Ng)
end

function print_status_anneal(mc, ig)
    bestval = maximum(mc.log_obj[:, 1:ig])

    println("-----------------------")
    println("iteration $ig")
    println("bestval = $bestval")
    println("-----------------------")
end

function demcz_anneal(logobj, Zmat, opts::DEMCopt; prevrun=nothing, temperaturefun::Function = tempbaseline, adaptγ = Dict("adapt"=>true,"minγ"=>0.1, "maxγ"=> 4.0, "adapt_every"=>500))
    return demcz_anneal(logobj, Zmat, opts.N, opts.K, opts.Ngeneration, opts.Nblocks, opts.blockindex, opts.eps_scale, opts.γ; prevrun=prevrun, verbose = opts.verbose, print_step=opts.print_step, temperaturefun = temperaturefun, T0 = opts.T0, TN = opts.TN, adaptγ=adaptγ)
end

function demcz_anneal(logobj, Zmat, N=4, K=10, Ngeneration=5000, Nblocks=1, blockindex=[1:size(Zmat,2)], eps_scale=1e-4*ones(size(Zmat,2)), γ=2.38; prevrun=nothing, verbose = true, print_step=100, temperaturefun::Function = tempbaseline, T0 = 3, TN = 0., adaptγ = Dict("adapt"=>true,"minγ"=>0.1, "maxγ"=> 4.0, "adapt_every"=>500))
    nrowZ, d = size(Zmat)
    Zmat = vcat(Zmat, zeros(Int(ceil(N*Ngeneration/K)), d))
    M = ones(Int64,1)*nrowZ
    if prevrun == nothing
        # start chain at last N elements of Z
        X = Zmat[end-N+1:end, :]
        # initial obj values
        log_objcurrent = map(logobj, [X[i,:] for i = 1:N])
    else
        # start chain at last generation of prevrun
        X = prevrun.chain[:,:,end]
        log_objcurrent = prevrun.log_objcurrent[:,end]
    end
    # preallocate the chain
    mc = MC(zeros(N,  d, Ngeneration),zeros(N, Ngeneration), X, log_objcurrent)
    # print inital values
    if verbose
        print_status(mc, 0)
    end
    # run through generations
    for ig = 1:Ngeneration
        for ic = 1:N
            runchain!(ic, ig, ig, mc, Zmat, K, M, logobj, blockindex, eps_scale, γ, Nblocks, temperaturefun)
        end
        if verbose
            if mod(ig, print_step) == 0.
                print_status_anneal(mc, to)
            end
        end
        if adaptγ["adapt"]
            if mod(ig, adaptγ["adapt_every"]) == 0.
                accept_ratio = sum(diff(mc.log_obj[:, ig- adaptγ["adapt_every"]+1:ig], dims=2).!=0.)/(N*adaptγ["adapt_every"])
                if accept_ratio < 0.1
                    γ = max(adaptγ["minγ"], γ*0.5)
                elseif accept_ratio >0.5
                    γ = min(adaptγ["maxγ"], γ*1.5)
                end
            end
        end
    end

    if prevrun != nothing
        return MC(cat(prevrun.chain, mc.chain, dims=3),cat(prevrun.log_obj, mc.log_obj, dims=2), mc.Xcurrent, mc.log_objcurrent), Zmat
    else
        return mc, Zmat
    end
end

function runchain!(ic, from, to, mc, Zmat, K, M, logobj, blockindex, eps_scale, γ, Nblocks, temperaturefun; T0 = 1, TN = 1e-3, Ngen = 1000)
    for ig = from:to
        Xcurrent, current_logobj = update_blocks(mc.Xcurrent[ic, :], mc.log_objcurrent[ic], Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks, temperaturefun(ig, Ngen, T0, TN))
        # update in chain
        mc.chain[ic, :, ig] .= Xcurrent
        mc.log_obj[ic, ig] = current_logobj
        mc.Xcurrent[ic, :] .= Xcurrent
        mc.log_objcurrent[ic] = current_logobj
        if mod(ig, K) == 0
            Zmat[M[1]+1,:] .= mc.Xcurrent[ic,:]
            M .= M .+ 1
        end
    end
end

function demcz_anneal_par(logobj, Zmat, opts::DEMCopt; prevrun=nothing, temperaturefun::Function = tempbaseline, sync_every=1000, adaptγ= Dict("adapt"=>true,"minγ"=>0.1, "maxγ"=> 4.0, "adapt_every"=>500))
    return demcz_anneal_par(logobj, Zmat, opts.N, opts.K, opts.Ngeneration, opts.Nblocks, opts.blockindex, opts.eps_scale, opts.γ; prevrun=prevrun,  temperaturefun = temperaturefun, T0 = opts.T0, TN = opts.TN, sync_every=sync_every, verbose = opts.verbose, adaptγ =adaptγ)
end

function demcz_anneal_par(logobj, Zmat, N=4, K=10, Ngeneration=5000, Nblocks=1, blockindex=[1:size(Zmat,2)], eps_scale=1e-4*ones(size(Zmat,2)), γ=2.38; prevrun=nothing, T0 = 3, TN = 1e-3, sync_every = 1000, temperaturefun::Function = tempbaseline, verbose=true, adaptγ = Dict("adapt"=>true,"minγ"=>0.1, "maxγ"=> 4.0, "adapt_every"=>500))
    # prep storage etc
    nrowZ, d = size(Zmat)
    global Zshared = SharedArray(vcat(Zmat, zeros(Int(ceil(N*Ngeneration/K)), d)))
    global M = SharedArray(ones(Int64,1)*nrowZ)
    if prevrun == nothing
        # start chain at last N elements of Z
        X = Zshared[M[1]-N+1:M[1], :]
        # initial obj values
        log_objcurrent = pmap(logobj, [X[i,:] for i = 1:N])
    else
        # start chain at last generation of prevrun
        X = prevrun.chain[:,:,end]
        log_objcurrent = prevrun.log_objcurrent[:,end]
    end

    # preallocate the chain
    global mc = MCShared(SharedArray(zeros(N,  d, Ngeneration)),SharedArray(zeros(N, Ngeneration)), SharedArray(X), SharedArray(log_objcurrent))
    global Ngenerationg = Ngeneration
    global T0g = T0
    global TNg = TN
    passobj(myid(), workers(), [:Zshared,:M, :Ngenerationg, :T0g, :TNg], from_mod=DEMC, to_mod=DEMC)
    passobj(myid(), workers(), [:mc], from_mod=DEMC, to_mod=DEMC)
    #passobj(myid(), workers(), [:temp], from_mod=DEMC, to_mod=DEMC)
    # each chain run through generations
    Nsets = ceil(Ngeneration/sync_every)
    splitgens = [(Int((i-1)*sync_every+1),Int(min(i*sync_every, Ngeneration)))  for i = 1:Nsets]
    global from = 0
    global to = 0
    global γpass = γ
    for set in splitgens
        global from = set[1]
        global to = set[2]
        passobj(myid(), workers(), [:from, :to, :γpass], from_mod=DEMC, to_mod=DEMC)
        pmap(ic -> runchain!(ic, from, to, mc, Zshared, K, M, logobj, blockindex, eps_scale, γpass, Nblocks, temperaturefun; T0=T0, TN=TN, Ngen=Ngeneration), 1:N)
        if verbose
            print_status_anneal(mc, to)
        end
        if adaptγ["adapt"]
            accept_ratio = sum(diff(mc.log_obj[:, from:to], dims=2).!=0.)/(N*(to-from))
            if accept_ratio < 0.1
                γpass = max(adaptγ["minγ"], γpass*0.5)
            elseif accept_ratio >0.5
                γpass = min(adaptγ["maxγ"], γpass*1.5)
            end
        end
    end

    if prevrun != nothing
        mc = MCShared(cat(prevrun.chain, mc.chain, dims=3),cat(prevrun.log_obj, mc.log_obj, dims=2), mc.Xcurrent, mc.log_objcurrent)
        return mc, Zshared
    else
        return mc, Zshared
    end
end

function update_blocks(Xcurrent, current_logobj, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks, temperature)
    for ib in 1:Nblocks
        Xcurrent, current_logobj = update_demcz_chain_block(Xcurrent, current_logobj, ib, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks, temperature)
    end
    return Xcurrent, current_logobj
end

function update_demcz_chain_block(Xcurrent, current_logobj, ib, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks, temperature)
    # generate proposal
    set = collect(1:M[1])
    i1 = rand(set)
    deleteat!(set, i1)
    i2 = rand(set)
    de_diffvec = zeros(eltype(Xcurrent), length(Xcurrent))
    block = blockindex[ib]
    blocklen = length(block)
    if blocklen == 1
        de_diffvec[block] = γ*(Zmat[i1,block]- Zmat[i2,block]) .+ eps_scale[block] .* randn()
    else
        de_diffvec[block] = γ/sqrt(2*blocklen)*(Zmat[i1,block]-Zmat[i2,block]) .+ eps_scale[block] .* randn(blocklen)
    end
    Xproposal = Xcurrent .+ de_diffvec
    log_objXprop = logobj(Xproposal)
    if accept(log_objXprop, current_logobj, temperature)
        return Xproposal, log_objXprop
    else
        return Xcurrent, current_logobj
    end
end

function accept(objprop, objprev, temperature)
    if log(rand()) < (objprop - objprev)/temperature
        return true
    else
        return false
    end
end
