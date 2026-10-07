using ParallelSolver

function getrange(N::Integer, myrank::Integer, commsize::Integer)
    q, r = divrem(N, commsize)
    start_idx = (myrank - 1) * q + min(myrank - 1, r) + 1
    len = q + (myrank <= r ? 1 : 0)
    return start_idx:(start_idx + len - 1)
end

MPI.Init()

n = 100
comm = MPI.COMM_WORLD

myrange = getrange(n^2, MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))

# Test Dot Product on Distributed Matrix Views
X_loc = rand(Float64, length(myrange), 3)
Y_loc = rand(Float64, length(myrange), 3)

X_dist = DistributedArray(X_loc, comm)
Y_dist = DistributedArray(Y_loc, comm)

v1 = view(X_dist, :, 1)
v2 = view(Y_dist, :, 2)

# Parallel dot product on SubArrays
dot_dist = dot(v1, v2)

# Global sequential reference
recvsizes = MPI.Gather(length(myrange), comm; root=0)
if MPI.Comm_rank(comm) == 0
    big_v1 = Vector{Float64}(undef, sum(recvsizes))
    big_v2 = Vector{Float64}(undef, sum(recvsizes))
    MPI.Gatherv!(v1.parent.loc[:, 1], VBuffer(big_v1, recvsizes), comm)
    MPI.Gatherv!(v2.parent.loc[:, 2], VBuffer(big_v2, recvsizes), comm)
    
    dot_seq = dot(big_v1, big_v2)
    println("Dot Product View Error: ", abs(dot_dist - dot_seq) / abs(dot_seq))
else
    MPI.Gatherv!(v1.parent.loc[:, 1], nothing, comm)
    MPI.Gatherv!(v2.parent.loc[:, 2], nothing, comm)
end
