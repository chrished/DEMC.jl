function demcz_sample(logobj, Zmat, N=4, K=10, Ngeneration=5000, Nblocks=1, blockindex=[1:size(Zmat,2)], eps_scale=1e-4*ones(size(Zmat,2)), γ=2.38; prevrun=nothing, verbose = true, print_step=100)
    M, d = size(Zmat)
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
            Xcurrent, current_logobj = update_blocks(mc.Xcurrent[ic, :], mc.log_objcurrent[ic], Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks)
            # update in chain
            mc.chain[ic, :, ig] .= Xcurrent
            mc.log_obj[ic, ig] = current_logobj
            mc.Xcurrent[ic, :] .= Xcurrent
            mc.log_objcurrent[ic] = current_logobj
        end
        if mod(ig, K) == 0.
            Zmat = vcat(Zmat, mc.Xcurrent)
            M += N
        end
        if verbose
            if mod(ig, print_step) == 0.
                print_status(mc, ig)
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
    avgpar = mean(mc.chain, dims = [1,3])
    avgval = mean(mc.log_obj, dims = [1,2])

    println("-----------------------")
    println("iteration $ig")
    println("average par = $avgpar")
    println("average val = $avgval")
    #println("bestval = $bestval")
    #println("bestpar = $bestpar")
    println("-----------------------")
end
#
# function demcz_sample_par(logobj, Zmat, N, K, Ngeneration, Nblocks, blockindex, eps_scale, γ; verbose=true, print_to_file=false, file = "./demcstat.txt")
#     wp = CachingPool(workers())
#     Mval, d = size(Zmat)
#     X = Zmat[end-N+1:end, :]
#     log_objcurrent = pmap(wp, logobj, [X[i,:] for i = 1:N])
#     mc = MC(Array{Float64}(N,  d, Ngeneration),Array{Float64}(N, Ngeneration), X, log_objcurrent)
#     global Xcurrent = copy(mc.Xcurrent)
#     global log_objcurrent = copy(mc.log_objcurrent)
#     global Z = Zmat
#     global M = Mval
#     @everywhere global Z
#     @everywhere global M
#     @everywhere global Xcurrent
#     @everywhere global log_objcurrent
#
#     passobj(myid(), workers(), [:Xcurrent, :log_objcurrent], from_mod = DEMC, to_mod = DEMC)
#     passobj(myid(), workers(), [:Z, :M], from_mod = DEMC, to_mod = DEMC)
#
#     if verbose
#         bestval = maximum(mc.log_objcurrent)
#         bestpar = mc.Xcurrent[findfirst(bestval.==mc.log_objcurrent), :]
#         if print_to_file == true
#             f = open(file, "w")
#             println(f, "iteration 0")
#             println(f, "bestval = $bestval")
#             println(f, "bestpar = $bestpar")
#             close(f)
#         else
#             println("iteration 0")
#             println("bestval = $bestval")
#             println("bestpar = $bestpar")
#         end
#     end
#     for ig = 1:Ngeneration
#         passobj(myid(), workers(), [:Xcurrent, :log_objcurrent], from_mod = DEMC, to_mod = DEMC)
#         res = pmap(wp, ic -> update_blocks(Xcurrent[ic,:], log_objcurrent[ic], Z, M, logobj, blockindex, eps_scale, γ, Nblocks), 1:N)
#         for ic = 1:N
#             # update in chain
#             mc.chain[ic, :, ig] = res[ic][1]
#             mc.log_obj[ic, ig] = res[ic][2]
#             mc.Xcurrent[ic, :] = res[ic][1]
#             mc.log_objcurrent[ic] = res[ic][2]
#             Xcurrent[ic, :] = res[ic][1]
#             log_objcurrent[ic] = res[ic][2]
#         end
#
#         if mod(ig, K) == 0.
#             Z = vcat(Z, mc.Xcurrent)
#             M += N
#             passobj(myid(), workers(), [:Z, :M], from_mod = DEMC, to_mod = DEMC)
#             if verbose
#                 bestval = maximum(mc.log_objcurrent)
#                 bestpar[:] = mc.Xcurrent[findfirst(bestval.==mc.log_objcurrent), :]
#                 avglast100 = mean(mc.chain[:, :, max(1, ig-100):ig], [1,3])
#                 avglast250 = mean(mc.chain[:, :, max(1, ig-250):ig], [1,3])
#                 accept_ratio = mean(mc.log_obj[:,max(1, ig-250):ig-1].!=mc.log_obj[:,max(2, ig-249):ig],2)
#                 accept_ratio100 = mean(mc.log_obj[:,max(1, ig-100):ig-1].!=mc.log_obj[:,max(2, ig-99):ig],2)
#                 if print_to_file
#                     f = open(file, "w")
#                     println(f, "iteration $ig")
#                     println(f, "bestval = $bestval")
#                     println(f, "bestpar = $bestpar")
#                     println(f, "avg last 100 = $avglast100")
#                     println(f, "accept_ratio last 100 = $accept_ratio100")
#                     println(f, "avg last 250 = $avglast250")
#                     println(f, "accept_ratio last 250 = $accept_ratio")
#                     close(f)
#                 else
#                     println("iteration $ig")
#                     println("bestval = $bestval")
#                     println("bestpar = $bestpar")
#                     println("avg last 100 = $avglast100")
#                     println("accept_ratio last 100 = $accept_ratio100")
#                     println("avg last 250 = $avglast250")
#                     println("accept_ratio last 250 = $accept_ratio")
#                 end
#             end
#         end
#     end
#     return mc
# end


function update_blocks(Xcurrent, current_logobj, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks)
    for ib in 1:Nblocks
        Xcurrent, current_logobj = update_demcz_chain_block(Xcurrent, current_logobj, ib, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks)
    end
    return Xcurrent, current_logobj
end


function update_demcz_chain_block(Xcurrent, current_logobj, ib, Zmat, M, logobj, blockindex, eps_scale, γ, Nblocks)
    # generate proposal
    set = collect(1:M)
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
