using Distributed
addprocs(2)
@everywhere using SharedArrays
@everywhere using ParallelDataTransfer

# setup Z and M
Z = SharedArray(vcat(randn(5,5), zeros(20*nworkers(),5)))
M = SharedArray(ones(Int64,1)*5)

# @distributed for col = 1:5
#     Z[:,col] .= myid()
# end

@everywhere function rungen!(Z,M)
    for ig = 1:100
        if mod(ig,5)==0
            println(M)
            println(Z[M[1]+1])
            Z[M[1]+1,:].= randn(5)
            M[1] = M[1] + 1
        end
    end
end

function parrun!(Z,M)
    pmap(i -> rungen!(Z,M), workers())
end


parrun!(Z,M)

rmprocs(workers())
