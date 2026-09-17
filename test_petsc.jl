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
    bicgstabl!(xj, Sj, bj, 1; reltol=1e-8, Pl=Si, log=false, verbose=false)
    # res_vec = DistributedVector(zeros(Float64, size(Sj.loc, 1)), Sj.comm)
    # mul!(res_vec, Sj, xj)
    # res_vec.loc .= bj.loc .- res_vec.loc

    # true_rel_res = norm(res_vec) / norm(bj)
    # if MPI.Comm_rank(Sj.comm) == 0
    #     println("True Relative Residual: ", true_rel_res)
    # end
end

function getrange(N::Integer, myrank::Integer, commsize::Integer)
    q, r = divrem(N, commsize)
    start_idx = (myrank - 1) * q + min(myrank - 1, r) + 1
    len = q + (myrank <= r ? 1 : 0)
    return start_idx:(start_idx + len - 1)
end

function Laplace_2D_5P_slice(n::Int, row_range::UnitRange{Int})
    N_global = n^2
    N_local  = length(row_range)
    
    # Pre-allocate COO arrays (approx. 5 non-zeros per row)
    nnz_est = 5 * N_local
    I = sizehint!(Int[], nnz_est)      # Local row indices: 1 .. N_local
    J = sizehint!(Int[], nnz_est)      # Global column indices: 1 .. N_global
    V = sizehint!(Float64[], nnz_est)  # Stencil values
    
    for (i_loc, i_glob) in enumerate(row_range)
        # Convert global row index to 2D grid coordinates (1-based)
        r = div(i_glob - 1, n) + 1
        c = mod(i_glob - 1, n) + 1
        
        # Center (diagonal)
        push!(I, i_loc)
        push!(J, i_glob)
        push!(V, 4.0)
        
        # Left neighbor
        if c > 1
            push!(I, i_loc)
            push!(J, i_glob - 1)
            push!(V, -1.0)
        end
        
        # Right neighbor
        if c < n
            push!(I, i_loc)
            push!(J, i_glob + 1)
            push!(V, -1.0)
        end
        
        # Bottom neighbor
        if r > 1
            push!(I, i_loc)
            push!(J, i_glob - n)
            push!(V, -1.0)
        end
        
        # Top neighbor
        if r < n
            push!(I, i_loc)
            push!(J, i_glob + n)
            push!(V, -1.0)
        end
    end
    
    return sparse(I, J, V, N_local, N_global)
end

MPI.Init()
comm = MPI.COMM_WORLD

# petsclib = PETSc.getlib(; PetscScalar=Float64, PetscInt=Int64)
# PETSc.initialize(petsclib, log_view=false)

myrank = MPI.Comm_rank(comm)+1
commsize  = MPI.Comm_size(comm)

n = 500
dims = n^2


myrange = getrange(n^2, myrank, commsize)
S = Laplace_2D_5P_slice(n, myrange)


A = DistributedMatrix(S, comm)
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
