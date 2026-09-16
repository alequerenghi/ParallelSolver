using BenchmarkTools
using SparseArrays
using MPI
using IncompleteLU
using IterativeSolvers
using PETSc

include("poisson_2D_matrices.jl")
include("distributed_matrix.jl")


function PETScsolve(xj::Vector, Sj::SparseMatrixCSC, bj::Vector, comm::MPI.Comm)
    ksp = PETSc.KSP(petsclib, comm, Sj; ksp_type="bcgs", pc_type="ilu", ksp_rtol=1e-8)

    b = PETSc.VecSeq(petsclib, bj)
    x = PETSc.VecSeq(petsclib, xj)

    PETSc.solve!(x, ksp, b)
    return x
end

function juliasolve(xj, Sj, bj)
    Si = ilu(Sj)
    bicgstabl!(xj, Sj, bj, 1; reltol=1e-8, Pl=Si, log=true, verbose=false)
end

function getrange(N::Integer, myrank::Integer, commsize::Integer)
    q, r = divrem(N, commsize)
    start_idx = (myrank - 1) * q + min(myrank - 1, r) + 1
    len = q + (myrank <= r ? 1 : 0)
    return start_idx:(start_idx + len - 1)
end

MPI.Init()
comm = MPI.COMM_WORLD

# petsclib = PETSc.getlib(; PetscScalar=Float64, PetscInt=Int64)
# PETSc.initialize(petsclib, log_view=false)

myrank = MPI.Comm_rank(comm)+1
commsize  = MPI.Comm_size(comm)

n = 1000
dims = n^2


S = nothing
for rank ∈ 1:commsize
    if rank == myrank
        global S
        S = Laplace_2D_5P(n)
        myrange = getrange(S.m, myrank, commsize)
        S = S[myrange, :]
    end
    MPI.Barrier(comm)
end

A = DistributedMatrix(S, comm)



b = DistributedVector(A.loc * randn(Float64, size(A, 1)), comm)
xj = similar(b)
xp = similar(b)

# t = @benchmark PETScsolve($xp, $S, $b, $comm) setup=fill!($xp, 0.0)
# display(t)

juliasolve(xj, A, b) # setup=fill!(xj, 0.0)

# display(tj)

# PETSc.finalize(petsclib)
