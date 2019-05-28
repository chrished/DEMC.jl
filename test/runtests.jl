using DEMC
using Test

@testset "DEMCz Sampling" begin
    # Write your own tests here.
    println("test with norm pdf in serial computation")
    include("example_normpdf.jl")
    println("test with norm pdf in parallel computation")
    include("example_normpdf_parallel.jl")
end

@testset "DEMCz Annealing" begin
    println("test annealing with quadratic optimization")
    include("test_anneal.jl")
    println("test annealing with quadratic optimization in parallel")
    include("test_anneal_parallel.jl")
end
