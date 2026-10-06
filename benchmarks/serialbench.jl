using IncompleteLU
using IterativeSolvers
using BenchmarkTools

include("../poisson_2D_matrices.jl")
include("../save_data.jl")

function test(A, b, x)
    Pl = ilu(A)
    bicgstabl!(x, A, b, 1; reltol=1e-8, Pl=Pl, log=false, verbose=false)
end

n = isempty(ARGS) ? 100 : parse(Int, ARGS[1])

A = Laplace_2D_9P(n)
b = A * rand(size(A,1))
x = similar(b)

bench = @benchmarkable test(A, b, x) setup=fill!(x, 0.0)
t = run(bench; evals=1, seconds=60, samples=100)

fsetup = "execution_time.csv"

savedata(fsetup, t, n, "poisson-9p")


