using BenchmarkTools
using SparseArrays
using MPI
using IncompleteLU
using IterativeSolvers
using PETSc

include("../poisson_2D_matrices.jl")
using ParallelSolver
include("../utilities.jl")

function juliasolve(xj, Sj, bj)
    # Si = ilu(Sj)
    Si = RASPreconditioner(Sj, 3)
    bicgstabl!(xj, Sj, bj, 2; reltol=1e-8, Pl=Si, log=true, verbose=iszero(MPI.Comm_rank(Sj.comm)))
    # res_vec = DistributedVector(zeros(Float64, size(Sj.loc, 1)), Sj.comm)
    # mul!(res_vec, Sj, xj)
    # res_vec.loc .= bj.loc .- res_vec.loc

    # true_rel_res = norm(res_vec) / norm(bj)
    # if MPI.Comm_rank(Sj.comm) == 0
    #     println("True Relative Residual: ", true_rel_res)
    # end
end

MPI.Init()
comm = MPI.COMM_WORLD

# petsclib = PETSc.getlib(; PetscScalar=Float64, PetscInt=Int64)
# PETSc.initialize(petsclib, log_view=false)

myrank = MPI.Comm_rank(comm)+1
commsize  = MPI.Comm_size(comm)

n = 1000
dims = n^2


myrange = getrange(n^2, myrank, commsize)


A = DistributedMatrix(Laplace_2D_5P_slice(n, myrange), comm)
# sendmaps = [Int[] for _ = 1:commsize]
# recvmaps = [Int[] for _ = 1:commsize]
# ghostmaps = Dict{Int, Int}()
# R = schwarz!(sendmaps, recvmaps, ghostmaps, A, 4)
# for rank in 1:commsize
#     if myrank == rank
#         println()
#         show(stdout, "text/plain", R)
#         println()
#     end
#     MPI.Barrier(comm)
# end

b = DistributedVector(A.loc * randn(Float64, size(A,1)), comm)
xj = zeros(length(b))

juliasolve(xj, A, b)

