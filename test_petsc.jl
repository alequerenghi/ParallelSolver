using BenchmarkTools
using SparseArrays
using MPI
using IncompleteLU
using IterativeSolvers
using PETSc

include("poisson_2D_matrices.jl")
include("distributed_matrix.jl")
include("utilities.jl")


function PETScsolve(xj::Vector, Sj::SparseMatrixCSC, bj::Vector, comm::MPI.Comm)
    ksp = PETSc.KSP(petsclib, comm, Sj; ksp_type="bcgs", pc_type="ilu", ksp_rtol=1e-8)

    b = PETSc.VecSeq(petsclib, bj)
    x = PETSc.VecSeq(petsclib, xj)

    PETSc.solve!(x, ksp, b)
    return x
end

function juliasolve(xj, Sj, bj)
    Si = ilu(Sj, τ=1e-3)
    bicgstabl!(xj, Sj, bj, 1; reltol=1e-8, Pl=Si, log=true, verbose=iszero(MPI.Comm_rank(Sj.comm)))
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
println("assembled!")



b = DistributedVector(A.loc * randn(Float64, size(A, 1)), comm)
xj = similar(b)
xp = similar(b)

# t = @benchmark PETScsolve($xp, $S, $b, $comm) setup=fill!($xp, 0.0)
# display(t)

bench = @benchmarkable juliasolve($xj, $A, $b) setup=fill!(xj, 0.0)
t = run(bench; evals=1, seconds=60, samples=100)

# display(tj)

# PETSc.finalize(petsclib)
