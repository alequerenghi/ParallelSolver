using ParallelSolver
using IncompleteLU
using IterativeSolvers
using MPI
using Random: seed!

include("../poisson_2D_matrices.jl")
include("../utilities.jl")
include("../save_data.jl")

seed!(42)

function @main(args; maxseconds=6000)
    MPI.Init()

    comm = MPI.COMM_WORLD
    commsize = MPI.Comm_size(comm)
    myrank = MPI.Comm_rank(comm)+1

    n = parse(Int, args[1])
    overlap = parse(Int, args[2])

    myrange = getrange(n^2, myrank, commsize)
    A = Laplace_2D_9P(n, myrange)
    D = DistributedMatrix(A, comm)
    b = DistributedVector(A * rand(size(A, 2)), comm)
    x = similar(b)

    τ = 4e-2
    Pl = (1 > overlap) ? Pl = ilu(D, τ=τ) : RASPreconditioner(D, overlap; τ=τ)

    iterations = Int[]
    starttime = time_ns()
    maxtime   = UInt64(maxseconds * 1e9)

    while maxtime > (time_ns()-starttime) && 100 > length(iterations)
        iters=0
        fill!(x.loc, 0.0)
        iterable = IterativeSolvers.bicgstabl_iterator!(
            x,
            D,
            b,
            1;
            Pl = Pl,
            reltol = 1e-8,
            initial_zero = true,
        )
        for _ in iterable
            iters+=1
        end
        push!(iterations, iters)
    end

    savedata("iterations.csv", iterations, n, "poisson-9p", comm, overlap)
    return nothing
end
