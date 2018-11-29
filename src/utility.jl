
# R̂ of Gelman: page 284
function Rhat_gelman(chain, Npop, Ngeneration, Npar)
    # split each chain in two
    n = Int(floor(Ngeneration/2))
    m = Npop*2
    #chain_split = vcat(chain[:,:,1:n], chain[:,:,n+1:end])
    chainsplit = zeros(eltype(chain), (m, Npar, n))
    chainsplit[1:Npop, :, :] .= chain[:,:,1:n]
    chainsplit[Npop+1:end, :, :] .= chain[:,:,n+1:2*n]
    avg_par = mean(chainsplit, [1,3])
    avg_chains = mean(chainsplit, 3)

    B = n/(m-1) * sum((avg_chains .- avg_par).^2, 1)
    sj = 1/(n-1)*sum((chainsplit .- avg_chains).^2, 3)
    W = 1/m * sum(sj, 1)
    varhat = (n-1)/n * W + 1/n*B
    Rhat =zeros(Npar)
    Rhat[:] = sqrt.(varhat./W)[:]
    return Rhat
end

function flatten_chain(chain, Npop, Ngeneration, Npar)
    flatchain = Array{eltype(chain)}(Npar, Npop*Ngeneration)
    for i = 1:Npar
        count = 0
        for ic = 1:Npop, ig = 1:Ngeneration
            count += 1
            flatchain[i, count] = chain[ic, i, ig]
        end
    end
    return flatchain
end

function convergence_check(chain, log_obj, Npop, Ngeneration, Npar, figure_path; verbose = true, parnames = [])
    nrow, ncol = size(log_obj)
    @assert nrow == Npop "number of rows of log_obj is not equal to number of chains"
    @assert ncol == Ngeneration "number of columns of log_obj is not equal to number of generations simulated"
    n1, n2, n3 = size(chain)
    @assert n1 == Npop "#elements in first dimension of chain is not equal to number of chains simulated"
    @assert n2 == Npar "#elements in second dimension is not equal to number of parameters"
    @assert n3 == Ngeneration "#elements in third dimension of chain is not equal to number of generations simulated"

    if parnames == []
        for ip = 1:Npar
            append!(parnames, [string("par ", ip)])
        end
    end
    if verbose
        try
            mkdir(string(figure_path))
        catch e
            println(e)
        end
    end
    # acceptance ratio
    accept_ratio = sum(diff(log_obj, 2).!=0., 2)./(Ngeneration-1)
    # plot the trace of the obj function value of the chains, clean out extreme values
    if verbose
        p_trace = plot(log_obj')
        savefig(p_trace, string(figure_path, "trace_logobj.png"))

        traceplots = []
        for par in 1:Npar
            append!(traceplots, [plot(reshape(chain[:, par , :], Npop, Ngeneration)', title=parnames[par])])
        end
        for par in 1:Npar
            par_trace = plot(traceplots[par])
            savefig(par_trace, string(figure_path, "trace_par_", par, ".png"))
        end
        for par in 1:Npar
            hi = histogram(chain[:, par , :][:], nbins = 33, normed=true)
            savefig(hi, string(figure_path, "hist_par_", par, ".png"))
        end
    end

    # Rhat Gelman
    Rhat = Rhat_gelman(chain, Npop, Ngeneration, Npar)
    if verbose
        println("Summary Checks")
        println(" ")
        println("Acceptance Ratio of each chain:")
        println(accept_ratio)
        println(" ")
        println("Rhat Gelman: $Rhat")
        println(" ")
        display(p_trace)
    end
    return accept_ratio, Rhat
end

function mean_cov_chain(chain, Npop, Ngeneration, Npar)
    n1, n2, n3 = size(chain)
    @assert n1 == Npop "#elements in first dimension of chain is not equal to number of chains simulated"
    @assert n2 == Npar "#elements in second dimension is not equal to number of parameters"
    @assert n3 == Ngeneration "#elements in third dimension of chain is not equal to number of generations simulated"

    flatchain = flatten_chain(chain, Npop, Ngeneration, Npar)
    b = mean(flatchain, 2)[:]
    cov = 1./(Ngeneration*Npop) * (flatchain .- b) * (flatchain .- b)'
    return b, cov
end

function save_res(b, Σb, Rhat, accept_ratio, path)
    writecsv(string(path, "_b.csv"), b)
    writecsv(string(path, "_Sigmab.csv"), Σb)
    writecsv(string(path, "_Rhat.csv"), Rhat)
    writecsv(string(path, "_accept_ratio.csv"), accept_ratio)
end

function extract_best(mc, N)
    bestval = maximum(mc.log_obj)
    (rows, cols) = findn(bestval.==mc.log_obj)
    bestpar = mc.chain[rows[1], :, cols[1]]
    return bestval, bestpar
end
