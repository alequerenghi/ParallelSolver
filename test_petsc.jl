using BenchmarkTools
using SparseArrays
using MPI
using IncompleteLU
using IterativeSolvers
using PETSc

function PETScsolve(xj::Vector, Sj::SparseMatrixCSC, bj::Vector, comm::MPI.Comm)
    ksp = PETSc.KSP(petsclib, comm, Sj; ksp_type="bcgs", pc_type="ilu", ksp_rtol=1e-8)

    b = PETSc.VecSeq(petsclib, bj)
    x = PETSc.VecSeq(petsclib, xj)

    PETSc.solve!(x, ksp, b)
    return x
end

function juliasolve(xj, Sj, bj)
    Si = ilu(Sj, τ=0.0)
    bicgstabl!(xj, Sj, bj, 1; reltol=1e-8, Pl=Si)
end


MPI.Init()
comm = MPI.COMM_SELF

petsclib = PETSc.getlib(; PetscScalar=Float64, PetscInt=Int64)
PETSc.initialize(petsclib, log_view=false)

include("poisson_2D_matrices.jl")

n = 500
dims = n^2

S = Laplace_2D_5P(n)
b = S * randn(Float64, size(S, 1))
xj = similar(b)
xp = similar(b)

# t = @benchmark PETScsolve($xp, $S, $b, $comm) setup=fill!($xp, 0.0)
# display(t)

tj = @benchmark juliasolve($xj, $S, $b) setup=fill!($xj, 0.0)

display(tj)

PETSc.finalize(petsclib)
