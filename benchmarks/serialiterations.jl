using IterativeSolvers
using IncompleteLU

include("../poisson_2D_matrices.jl")
include("../save_data.jl")

n = parse(Int, ARGS[1])

A = Laplace_2D_9P(n)
println(size(A))
b = A * rand(size(A, 2))
x = similar(b)

Pl = ilu(A)

iterations = Int[]

for _ = 1:20
    iters=0
    fill!(x, 0.0)
    iterable =
        IterativeSolvers.bicgstabl_iterator!(x, A, b, 1; Pl = Pl, reltol = 1e-8)
    for _ in iterable
        iters+=1
    end
    push!(iterations, iters)
end


savedata("iterations.csv", iterations, n, "poisson-9p")
