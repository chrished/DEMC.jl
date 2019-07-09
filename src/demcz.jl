function demcz_sample(logobj, Zmat, opts::DEMCopt; prevrun=nothing)
    return demcz_sample(logobj, Zmat, opts.N, opts.K, opts.Ngeneration, opts.Nblocks, opts.blockindex, opts.eps_scale, opts.γ; prevrun=prevrun, verbose = opts.verbose, print_step=opts.print_step, autostop=opts.autostop, autostop_Rhat=opts.autostop_Rhat, autostop_every = opts.autostop_every)
end
"""
demcz_sample(logobj, Zmat, N=4, K=10, Ngeneration=5000, Nblocks=1, blockindex=[1:size(Zmat,2)], eps_scale=1e-4*ones(size(Zmat,2)), γ=2.38; prevrun=nothing, verbose = true, print_step=100)

Serial run of DEMC chain. At each generation all chains are updated one after the other.
"""
function demcz_sample(logobj, Zmat, N=4, K=10, Ngeneration=5000, Nblocks=1, blockindex=[1:size(Zmat,2)], eps_scale=1e-4*ones(size(Zmat,2)), γ=2.38; prevrun=nothing, verbose = true, print_step=100, autostop=:no, autostop_Rhat=1.01, autostop_every = 1000)
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
            runchain!(ic, ig, ig, mc, Zmat, K, M, logobj, blockindex, eps_scale, γ, Nblocks)
        end
        if verbose
            if mod(ig, print_step) == 0.
                print_status(mc, ig)
            end
        end
        if autostop == :Rhat
            if mod(ig, autostop_every) == 0
                Rhat = Rhat_gelman(mc.chain[:,:, ig-autostop_every+1:ig], N, autostop_every, d)
                accept_ratio = sum(diff(mc.log_obj[ig-autostop_every+1:ig], dims = 2).!=0., dims = 2)./ (autostop_every-1)
                if (maximum(Rhat) < autostop_Rhat)
                    if mean(accept_ratio) < 0.1
                        println("Warning: accept ratio below 10% on average")
                    end
                    mc = MC(mc.chain[:,:,1:ig],  mc.log_obj[:, 1:ig], mc.Xcurrent, mc.log_objcurrent)
                    if prevrun != nothing
                        return MC(cat(prevrun.chain, mc.chain, dims=3),cat(prevrun.log_obj, mc.log_obj, dims=2), mc.Xcurrent, mc.log_objcurrent), Zmat
                    else
                        return mc, Zmat[1:M[1],:]
                    end
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

function print_status(mc, ig)
    #bestval = maximum(mc.log_objcurrent)
    #bestpar = mc.Xcurrent[findfirst(bestval.==mc.log_objcurrent), :]
    avgpar = mean(mc.chain[:,:,1:max(1,ig)], dims = [1,3])
    avgval = mean(mc.log_obj[:,1:max(1,ig)], dims = [1,2])

    println("-----------------------")
    println("iteration $ig")
    println("average par = $avgpar")
    println("average val = $avgval")
    #println("bestval = $bestval")
    #println("bestpar = $bestpar")
    println("-----------------------")
end

function runchain!(ic, from, to, mc, Zmat, K, M, logobj, blockindex, eps_scale, γ, Nblocks)
    for ig = from:to
        Xcurrent, current_logobj = update_blocks(mc.Xcurrent[ic, :], mc.log_objcurrent[ic], Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks)
        # update in chain
        mc.chain[ic, :, ig] .= Xcurrent
        mc.log_obj[ic, ig] = current_logobj
        mc.Xcurrent[ic, :] .= Xcurrent
        mc.log_objcurrent[ic] = current_logobj
        if mod(ig, K) == 0.
            Zmat[M[1]+1,:] .= mc.Xcurrent[ic,:]
            M .= M .+ 1
        end
    end
end


"""
demcz_sample_par(logobj, Zmat, opts::DEMCopt; sync_every = 1000, prevrun=nothing)

Runs each chain on a separate process - Z is updated simultaenously among all chains running in parallel.
"""
function demcz_sample_par(logobj, Zmat, opts::DEMCopt; sync_every = 1000, prevrun=nothing)
    return demcz_sample_par(logobj, Zmat, opts.N, opts.K, opts.Ngeneration, opts.Nblocks, opts.blockindex, opts.eps_scale, opts.γ;sync_every = sync_every, prevrun=prevrun,  autostop=opts.autostop, autostop_Rhat=opts.autostop_Rhat, verbose=opts.verbose)
end


function demcz_sample_par(logobj, Zmat, N=4, K=10, Ngeneration=5000, Nblocks=1, blockindex=[1:size(Zmat,2)], eps_scale=1e-4*ones(size(Zmat,2)), γ=2.38; sync_every = 1000, prevrun=nothing, autostop=:Rhat, autostop_Rhat=1.1, verbose=true)
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
    # define on all cores
    passobj(myid(), workers(), [:Zshared,:M, :Ngenerationg], from_mod=DEMC, to_mod=DEMC)
    passobj(myid(), workers(), [:mc], from_mod=DEMC, to_mod=DEMC)
    # split up generations
    Nsets = ceil(Ngeneration/sync_every)
    splitgens = [(Int((i-1)*sync_every+1),Int(min(i*sync_every, Ngeneration)))  for i = 1:Nsets]
    global from = 0
    global to = 0
    for set in splitgens
        global from = set[1]
        global to = set[2]
        passobj(myid(), workers(), [:from, :to], from_mod=DEMC, to_mod=DEMC)
        pmap(ic -> runchain!(ic, from, to, mc, Zshared, K, M, logobj, blockindex, eps_scale, γ, Nblocks), 1:N)
        if verbose
            print_status(mc, to)
        end
        if autostop == :Rhat
            Rhat = Rhat_gelman(mc.chain[:,:, from:to], N, to-from, d)
            accept_ratio = sum(diff(mc.log_obj, dims = 2).!=0., dims = 2)./ (sync_every-1)
            if (maximum(Rhat) < autostop_Rhat)
                if mean(accept_ratio) < 0.1
                    println("Warning: accept ratio below 10% on average")
                end
                mc = MC(mc.chain[:,:,1:to],  mc.log_obj[:, 1:to], mc.Xcurrent, mc.log_objcurrent)
                if prevrun != nothing
                    mc = MCShared(cat(prevrun.chain, mc.chain, dims=3),cat(prevrun.log_obj, mc.log_obj, dims=2), mc.Xcurrent, mc.log_objcurrent)
                    return mc, Zshared[1:M[1],:]
                else
                    return mc, Zshared[1:M[1],:]
                end
            end
        end
    end

    if prevrun != nothing
        mc = MCShared(cat(prevrun.chain, mc.chain, dims=3),cat(prevrun.log_obj, mc.log_obj, dims=2), mc.Xcurrent, mc.log_objcurrent)
        return mc, Zmat
    else
        return mc, Zmat
    end
end

function update_blocks(Xcurrent, current_logobj, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks)
    for ib in 1:Nblocks
        Xcurrent, current_logobj = update_demcz_chain_block(Xcurrent, current_logobj, ib, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks)
    end
    return Xcurrent, current_logobj
end

function update_demcz_chain_block(Xcurrent, current_logobj, ib, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks)
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
    if accept(log_objXprop, current_logobj)
        return Xproposal, log_objXprop
    else
        return Xcurrent, current_logobj
    end
end

function accept(objprop, objprev)
    if log(rand()) < objprop - objprev
        return true
    else
        return false
    end
end
