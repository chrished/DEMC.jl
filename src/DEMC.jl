module DEMC
    using LinearAlgebra
    using Statistics
    using Plots
    gr()

    include("demcz.jl")
    include("utils.jl")

    struct MC
        chain::Array{Float64, 3} # parameter population for all generations
        log_obj::Array{Float64, 2} # log obj along the chain
        Xcurrent::Array{Float64, 2} # population
        log_objcurrent::Array{Float64, 1} # log obj values
    end


end # module
