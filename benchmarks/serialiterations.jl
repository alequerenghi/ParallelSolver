using IterativeSolvers
using IncompleteLU

include("../poisson_2D_matrices.jl")



A = Laplace_2D_9P(n)
b = rand(size(A,2))
x = similar(b)

iterable = IterativeSolvers.bicgstabl_iterator!(x, A, b, 1; Pl=Pl, reltol=1e-8, verbose=false, log=false)

iterations = Int[]

for _ in 1:20
    iters=0
for _ in iterable
    iters+=1
end
push!(iterations, iters)
end


    
