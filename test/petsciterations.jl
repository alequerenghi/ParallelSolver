include("gemini_petsc.jl")

function @main(args)
    n = parse(Int, args[1])
    overlap = parse(Int, args[2])
    N = n^2
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
    petsclib = first(PETSc.petsclibs)
    comm = MPI.COMM_WORLD
    PETSc.initialize(petsclib; options=petscargs)
    PetscScalar = PETSc.scalartype(petsclib)
    PetscInt    = PETSc.inttype(petsclib)
    myrange = getrange(N, MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))
    S = Laplace_2D_9P(n, myrange)
    A = assemble!(petsclib, S, comm)
    bdata = S * rand(size(S,2))
    b = assemble!(petsclib, bdata, comm, N)
    x = similar(b)

    iterations = testrun(A, x, b, petsclib, comm)

    savedata("petsc_iterations.csv", iterations, n, "poisson-9p")

    PETSc.destroy!(A)
    PETSc.destroy!(b)
    PETSc.destroy!(x)

    PETSc.finalize(petsclib)
end

function testrun(A, x, b, petsclib, comm; maxseconds=6000)
    timelimit = UInt64(maxseconds*1e9)
    ksp = PETSc.KSP(A)
    LibPETSc.KSPSetFromOptions(petsclib, ksp)
    LibPETSc.KSPSetUp(petsclib, ksp)
    iterations = Int[]

    starttime = time_ns()
    while timelimit > (time_ns() - starttime) && 100 > length(iterations)
        fill!(x, 0.0)
        PETSc.solve!(x, ksp, b)
        push!(iterations, PETSc.iteration_number(ksp))
        MPI.Barrier(comm)
    end
    PETSc.destroy!(ksp)
    return iterations
end

