module DEMC
    using LinearAlgebra
    using Statistics
    using Plots
    using Distributed
    using SharedArrays
    using ParallelDataTransfer
    gr()

    include("demcz.jl")
    include("utils.jl")
    include("demcz_anneal.jl")

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

end # module
