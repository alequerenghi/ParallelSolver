using ParallelSolver
include("utilities.jl")
include("poisson_2D_matrices.jl")

MPI.Init()


comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)+1
commsize = MPI.Comm_size(comm)

# Compare Parallel vs Sequential Unpreconditioned BiCGStab(1)
n = 20
myrange = getrange(n^2, rank, commsize)
A_slice = Laplace_2D_5P_slice(n, myrange)
D = DistributedMatrix(A_slice, comm)

b_dist = DistributedVector(ones(Float64, size(D, 1)), comm)
x_dist = DistributedVector(zeros(Float64, size(D, 1)), comm)

# Solve in Parallel without Preconditioner
bicgstabl!(x_dist, D, b_dist, 1; reltol=1e-8, Pl=Identity(), initial_zero=true)

# Gather and Compare with Sequential Solve
recvsizes = MPI.Gather(length(x_dist.loc), comm; root=0)
if MPI.Comm_rank(comm) == 0
    bigx_parallel = Vector{Float64}(undef, sum(recvsizes))
    MPI.Gatherv!(x_dist.loc, VBuffer(bigx_parallel, recvsizes), comm; root=0)

    A_seq = Laplace_2D_5P(n)
    b_seq = ones(Float64, n^2)
    x_seq = zeros(Float64, n^2)
    
    bicgstabl!(x_seq, A_seq, b_seq, 1; reltol=1e-8, Pl=Identity(), initial_zero=true)
    
    sol_err = norm(bigx_parallel - x_seq) / norm(x_seq)
    println("Unpreconditioned Solver Solution Error vs Sequential: ", sol_err)
else
    MPI.Gatherv!(x_dist.loc, nothing, comm; root=0)
end
