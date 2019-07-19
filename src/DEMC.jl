module DEMC
    using LinearAlgebra
    using Statistics
    # using Plots
    using Distributed
    using SharedArrays
    using ParallelDataTransfer
    #gr()

    struct MC
        chain::Array{Float64, 3} # parameter population for all generations
        log_obj::Array{Float64, 2} # log obj along the chain
        Xcurrent::Array{Float64, 2} # population
        log_objcurrent::Array{Float64, 1} # log obj values
    end

    struct MCShared
        chain::SharedArray{Float64, 3} # parameter population for all generations
        log_obj::SharedArray{Float64, 2} # log obj along the chain
        Xcurrent::SharedArray{Float64, 2} # population
        log_objcurrent::SharedArray{Float64, 1} # log obj values
    end

    mutable struct DEMCopt
        N::Int
        K::Int
        Ngeneration::Int
        Nblocks::Int
        blockindex::Array
        eps_scale::Array{Float64, 1}
        γ::Float64
        verbose::Bool
        print_step::Int
        T0::Float64
        TN::Float64
        autostop::Symbol
        autostop_every::Int
        autostop_Rhat
    end

    function demcopt(Npar; N=4, K=10, Ngeneration=5000, Nblocks=1, blockindex=[1:Npar], eps_scale=1e-4*ones(Npar), γ=2.38, verbose = true, print_step=100, T0 = 3, TN = 1e-3, autostop = :Rhat, autostop_every = 1000, autostop_Rhat=1.05)
        return DEMCopt(N, K, Ngeneration, Nblocks, blockindex, eps_scale, γ, verbose, print_step, T0, TN, autostop, autostop_every, autostop_Rhat)
    end


    include("demcz.jl")
    include("utils.jl")
    include("demcz_anneal.jl")

end # module
