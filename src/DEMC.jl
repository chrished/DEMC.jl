module DEMC
    export demcz_sample, demcz_sample_par,  Rhat_gelman, convergence_check, mean_cov_chain, save_res, extract_best

    using ParallelDataTransfer
    using Plots

    struct MC
        chain::Array{Float64} # parameter population for all generations
        log_obj::Array{Float64} # log obj along the chain
        Xcurrent::Array{Float64} # population
        log_objcurrent::Array{Float64} # log obj values
        #accept::Array{Int64} #  number of acceptance in each subchain
    end

    struct HelperArrays
        chainset::Array{Int64}
        de_diffvec::Array{Float64}
    end

    ENV["GKSwstype"]="100"
    gr()
    plot(ones(3))

    include("demc06.jl") # algorithm from 2006 paper
    include("demc08.jl") # DEMCz(s) algorithms from 2008 paper
    include("demc08_anneal.jl") # Annealing variants of DEMCz(s) algorithms from 2008 paper
    include("utility.jl")

end # module
