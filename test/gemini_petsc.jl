using BenchmarkTools
using SparseArrays
using MPI
using PETSc
using Random: seed!

include("../utilities.jl")
include("../poisson_2D_matrices.jl")
include("../save_data.jl")

seed!(42)

overlap = parse(Int, ARGS[2])

if !MPI.Initialized()
    MPI.Init()
end

args = [
    "-ksp_type", "bcgsl",
    "-ksp_bcgsl_ell", "1",
    "-pc_type", "asm",
    "-pc_asm_overlap", "$overlap",          # Overlap width (2 grid point)
    "-pc_asm_type", "restrict",       # "restrict" = RAS, "basic" = Standard ASM
     "-sub_pc_type", "ilu",            # Local preconditioner on each rank
     "-sub_pc_factor_levels", "4",     # ILU(0)
    "-ksp_rtol", "1e-8",
     "-ksp_view"
]

comm = MPI.COMM_WORLD
petsclib = first(PETSc.petsclibs)
PETSc.initialize(petsclib; options=args)

PetscScalar = PETSc.scalartype(petsclib)
PetscInt    = PETSc.inttype(petsclib)

# ── Problem Setup (2D Poisson grid: n x n) ─────────────────────────
n = parse(Int, ARGS[1])
N = n^2 # Total global system size

# ── 1. Create MPI Parallel Matrix (A) ────────────────────────────────
A = LibPETSc.MatCreate(petsclib, comm)
LibPETSc.MatSetSizes(petsclib, A, PetscInt(LibPETSc.PETSC_DECIDE), PetscInt(LibPETSc.PETSC_DECIDE), PetscInt(N), PetscInt(N))
# LibPETSc.MatSetType(petsclib, A, "mpiaij") # Automatically picks MATMPIAIJ for size(comm) > 1
LibPETSc.MatSetUp(petsclib, A)
rstart, rend = LibPETSc.MatGetOwnershipRange(petsclib, A)

myrange = getrange(N, MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))
S = Laplace_2D_9P(n, myrange)
I, J, V = findnz(S)
nelements = length(S.nzval)
LibPETSc.MatSetPreallocationCOO(petsclib, A, PetscInt(nelements), PetscInt.(I .-1 .+ rstart), PetscInt.(J .-1))
LibPETSc.MatSetValuesCOO(petsclib, A, PetscScalar.(V), LibPETSc.INSERT_VALUES)

PETSc.assemble!(A)

# ── 3. Create Parallel RHS (b) and Solution (x) Vectors ─────────────
b = LibPETSc.VecCreate(petsclib, comm)
# b = LibPETSc.VecCreateMPI(petsclib, comm, PetscInt(rend-rstart), PetscInt(N))
LibPETSc.VecSetSizes(petsclib, b, PetscInt(LibPETSc.PETSC_DECIDE), PetscInt(N))
LibPETSc.VecSetFromOptions(petsclib, b)
LibPETSc.VecSetUp(petsclib, b)


bdata = S * rand(size(S,2))
LibPETSc.VecSetValues(petsclib, b, PetscInt(length(bdata)), PetscInt.(rstart:rend), PetscScalar.(bdata), LibPETSc.INSERT_VALUES)
PETSc.assemble!(b)

x = similar(b)

# Get 0-indexed row range owned by this MPI rank: [rstart, rend)
# 
# # ── 2. Assemble Matrix Entries locally ──────────────────────────────
# for row in rstart:(rend - 1)
#     i = div(row, n) # Row coordinate in 2D mesh
#     j = rem(row, n) # Column coordinate in 2D mesh
# 
#     # Diagonal entry
#     LibPETSc.MatSetValues(petsclib, A, PetscInt(1), [PetscInt(row)], PetscInt(1), [PetscInt(row)], PetscScalar[4.0], LibPETSc.INSERT_VALUES)
# 
#     # 5-point stencil off-diagonals
#     if j > 0     ; LibPETSc.MatSetValues(petsclib, A, PetscInt(1), [PetscInt(row)], PetscInt(1), [PetscInt(row - 1)], PetscScalar[-1.0], LibPETSc.INSERT_VALUES); end
#     if j < n - 1 ; LibPETSc.MatSetValues(petsclib, A, PetscInt(1), [PetscInt(row)], PetscInt(1), [PetscInt(row + 1)], PetscScalar[-1.0], LibPETSc.INSERT_VALUES); end
#     if i > 0     ; LibPETSc.MatSetValues(petsclib, A, PetscInt(1), [PetscInt(row)], PetscInt(1), [PetscInt(row - n)], PetscScalar[-1.0], LibPETSc.INSERT_VALUES); end
#     if i < n - 1 ; LibPETSc.MatSetValues(petsclib, A, PetscInt(1), [PetscInt(row)], PetscInt(1), [PetscInt(row + n)], PetscScalar[-1.0], LibPETSc.INSERT_VALUES); end
# end


function testrun(memusage, A, x, b, petsclib, comm)
    # ── 4. Setup KSP Parallel Solver & Solve ─────────────────────────────
    ksp = PETSc.KSP(A)
    LibPETSc.KSPSetFromOptions(petsclib, ksp)
    LibPETSc.KSPSetUp(petsclib, ksp)

    PETSc.solve!(x, ksp, b)
    push!(memusage, Sys.maxrss())
    PETSc.destroy!(ksp)
end

startrss = Sys.maxrss()
println(startrss/2^20)

memoryusage = UInt64[]

bench = @benchmarkable testrun($memoryusage, $A, $x, $b, $petsclib, $comm) setup=fill!(x, 0.0)
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
