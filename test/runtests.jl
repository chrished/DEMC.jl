using DEMC
using Test

@testset "DEMC.jl" begin
    # Write your own tests here.
    println("test with norm pdf in serial computation")
    include("example_normpdf.jl")
    println("test with norm pdf in parallel computation")
    include("example_normpdf_parallel.jl")
end
