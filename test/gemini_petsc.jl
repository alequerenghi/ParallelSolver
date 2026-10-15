using BenchmarkTools
using SparseArrays
using MPI
using PETSc
using Random: seed!

include("../utilities.jl")
include("../poisson_2D_matrices.jl")
include("../save_data.jl")

seed!(42)

function assemble!(lib, S::SparseMatrixCSC, comm)
    N           = size(S, 2)
    PetscScalar = PETSc.scalartype(lib)
    PetscInt    = PETSc.inttype(lib)

    A = LibPETSc.MatCreate(lib, comm)
    LibPETSc.MatSetSizes(lib, A, PetscInt(LibPETSc.PETSC_DECIDE), PetscInt(LibPETSc.PETSC_DECIDE), PetscInt(N), PetscInt(N))
    LibPETSc.MatSetUp(lib, A)
    rstart, rend = LibPETSc.MatGetOwnershipRange(lib, A)

    I, J, V = findnz(S)
    nelements = length(S.nzval)
    LibPETSc.MatSetPreallocationCOO(lib, A, Int64(nelements), PetscInt.(I .-1 .+ rstart), PetscInt.(J .-1))
    LibPETSc.MatSetValuesCOO(lib, A, PetscScalar.(V), LibPETSc.INSERT_VALUES)

    PETSc.assemble!(A)
    return A
end

function assemble!(lib, bdata::Vector, comm, N)
    PetscInt    = PETSc.inttype(lib)
    PetscScalar = PETSc.scalartype(lib)

    b = LibPETSc.VecCreate(lib, comm)
    # b = LibPETSc.VecCreateMPI(lib, comm, PetscInt(rend-rstart), PetscInt(N))
    LibPETSc.VecSetSizes(lib, b, PetscInt(LibPETSc.PETSC_DECIDE), PetscInt(N))
    LibPETSc.VecSetFromOptions(lib, b)
    LibPETSc.VecSetUp(lib, b)
    rstart, rend = LibPETSc.VecGetOwnershipRange(lib, b)

    LibPETSc.VecSetValues(lib, b, PetscInt(length(bdata)), PetscInt.(rstart:rend), PetscScalar.(bdata), LibPETSc.INSERT_VALUES)
    PETSc.assemble!(b)
    return b
end

function testrun(memusage, A, x, b, petsclib, comm)
    ksp = PETSc.KSP(A)
    LibPETSc.KSPSetFromOptions(petsclib, ksp)
    LibPETSc.KSPSetUp(petsclib, ksp)

    PETSc.solve!(x, ksp, b)
    push!(memusage, Sys.maxrss())
    PETSc.destroy!(ksp)
end

function @main(args)

    overlap = parse(Int, args[2])
    MPI.Init()
    petscargs = [
        "-ksp_type", "bcgsl",
        "-ksp_bcgsl_ell", "1",
        "-pc_type", "asm",
        "-pc_asm_overlap", "$overlap",          # Overlap width (2 grid point)
        "-pc_asm_type", "restrict",       # "restrict" = RAS, "basic" = Standard ASM
         "-sub_pc_type", "ilu",            # Local preconditioner on each rank
         "-sub_pc_factor_levels", "4",     # ILU(0)
        "-ksp_rtol", "1e-8",
         # "-ksp_view",
         # "-ksp_monitor",
    ]
    comm = MPI.COMM_WORLD
    petsclib = first(PETSc.petsclibs)
    PETSc.initialize(petsclib; options=petscargs)
    PetscScalar = PETSc.scalartype(petsclib)
    PetscInt    = PETSc.inttype(petsclib)
    n = parse(Int, args[1])
    N = n^2 # Total global system size
    myrange = getrange(N, MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))
    S = Laplace_2D_9P(n, myrange)

    A = assemble!(petsclib, S, comm)
    bdata = S * rand(size(S,2))

    b = assemble!(petsclib, bdata, comm, N)
    x = similar(b)

    startrss = Sys.maxrss()

    memoryusage = UInt64[]

    bench = @benchmarkable testrun($memoryusage, $A, $x, $b, $petsclib, $comm) setup=fill!($x, 0.0)
    t = run(bench; samples=100, evals=1, seconds=10)

    memoryusage .-= startrss

    avgmem = round(Int, mean(memoryusage))
    t.memory = avgmem
    savedata("petsc.csv", t, n, "poisson-9p", comm, overlap)



    # ── Cleanup ──────────────────────────────────────────────────────────

    PETSc.destroy!(A)
    PETSc.destroy!(b)
    PETSc.destroy!(x)

    PETSc.finalize(petsclib)

end
