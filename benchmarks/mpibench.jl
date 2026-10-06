using ParallelSolver
using MPI
using IncompleteLU
using IterativeSolvers
using BenchmarkTools
using Random: seed!

seed!(42)

include("../poisson_2D_matrices.jl")
include("../save_data.jl")
include("../utilities.jl")

function test(A, b, x, overlap)
    D = DistributedMatrix(A, comm)
    if 1 > overlap
        Pl = ilu(D)
    else
        Pl = RASPreconditioner(D, overlap)
    end
    bicgstabl!(x, D, b, 1; reltol=1e-8, Pl=Pl, log=false, verbose=false)
end

function setup_only(A, b, x, overlap)
    D = DistributedMatrix(A, comm)
    if 1 > overlap
        Pl = ilu(D)
    else
        Pl = RASPreconditioner(D, overlap)
    end
    IterativeSolvers.bicgstabl_iterator!(x, D, b, 1; Pl=Pl, reltol=1e-8)
end


MPI.Init()

comm = MPI.COMM_WORLD

commsize = MPI.Comm_size(comm)
myrank   = MPI.Comm_rank(comm)+1

n = isempty(ARGS) ? 100 : parse(Int, ARGS[1])
overlap = parse(Int, ARGS[2])

myrange = getrange(n^2, myrank, commsize)

A = Laplace_2D_9P(n, myrange)
b = DistributedVector(A * rand(size(A,2)), comm)
x = similar(b)

bench = @benchmarkable test($A, $b, $x, $overlap) setup=fill!(x.loc, 0.0)
t = run(bench; evals=1, seconds=6000, samples=100)

ftotal = "execution_time.csv"

savedata(ftotal, t, n, "poisson-9p", comm, overlap)

bench = @benchmarkable setup_only($A, $b, $x, $overlap) setup=fill!(x.loc, 0.0)
t = run(bench; evals=1, seconds=60, samples=100)

fsetup = "setup_time.csv"

savedata(fsetup, t, n, "poisson-9p", comm, overlap)
