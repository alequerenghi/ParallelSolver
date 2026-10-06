using ParallelSolver
include("poisson_2D_matrices.jl")
include("utilities.jl")

MPI.Init()

comm = MPI.COMM_WORLD

rank = MPI.Comm_rank(comm)+1
commsize = MPI.Comm_size(comm)

# Compare Parallel vs Sequential Unpreconditioned BiCGStab(1)
n = 20
myrange = getrange(n^2, rank, commsize)
A_slice = Laplace_2D_5P_slice(n, myrange)
D = DistributedMatrix(A_slice, comm)

# Verify ldiv! on local ILU factor
Si = ilu(D) # Local ILU factor
v_test = DistributedVector(rand(Float64, size(D, 1)), comm)
v_copy = copy(v_test.loc)

ldiv!(Si, local_slice(v_test))

# Verify local solve A_loc \ v_copy
ref_solve = Si \ v_copy
println("Preconditioner ldiv! Error: ", norm(local_slice(v_test) - ref_solve))


