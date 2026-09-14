using MPI
using PETSc

if !MPI.Initialized()
    MPI.Init()
end

comm = MPI.COMM_WORLD
petsclib = first(PETSc.petsclibs)
PETSc.initialized(petsclib) || PETSc.initialize(petsclib)

PetscScalar = PETSc.scalartype(petsclib)
PetscInt    = PETSc.inttype(petsclib)

# ── Problem Setup (2D Poisson grid: n x n) ─────────────────────────
n = 100
N = n^2 # Total global system size

# ── 1. Create MPI Parallel Matrix (A) ────────────────────────────────
A = LibPETSc.MatCreate(petsclib, comm)
LibPETSc.MatSetSizes(petsclib, A, PetscInt(LibPETSc.PETSC_DECIDE), PetscInt(LibPETSc.PETSC_DECIDE), PetscInt(N), PetscInt(N))
# LibPETSc.MatSetType(petsclib, A, "mpiaij") # Automatically picks MATMPIAIJ for size(comm) > 1
LibPETSc.MatSetUp(petsclib, A)

# Get 0-indexed row range owned by this MPI rank: [rstart, rend)
rstart, rend = LibPETSc.MatGetOwnershipRange(petsclib, A)

# ── 2. Assemble Matrix Entries locally ──────────────────────────────
for row in rstart:(rend - 1)
    i = div(row, n) # Row coordinate in 2D mesh
    j = rem(row, n) # Column coordinate in 2D mesh

    # Diagonal entry
    LibPETSc.MatSetValues(petsclib, A, PetscInt(1), [PetscInt(row)], PetscInt(1), [PetscInt(row)], PetscScalar[4.0], LibPETSc.INSERT_VALUES)

    # 5-point stencil off-diagonals
    if j > 0     ; LibPETSc.MatSetValues(petsclib, A, PetscInt(1), [PetscInt(row)], PetscInt(1), [PetscInt(row - 1)], PetscScalar[-1.0], LibPETSc.INSERT_VALUES); end
    if j < n - 1 ; LibPETSc.MatSetValues(petsclib, A, PetscInt(1), [PetscInt(row)], PetscInt(1), [PetscInt(row + 1)], PetscScalar[-1.0], LibPETSc.INSERT_VALUES); end
    if i > 0     ; LibPETSc.MatSetValues(petsclib, A, PetscInt(1), [PetscInt(row)], PetscInt(1), [PetscInt(row - n)], PetscScalar[-1.0], LibPETSc.INSERT_VALUES); end
    if i < n - 1 ; LibPETSc.MatSetValues(petsclib, A, PetscInt(1), [PetscInt(row)], PetscInt(1), [PetscInt(row + n)], PetscScalar[-1.0], LibPETSc.INSERT_VALUES); end
end

PETSc.assemble!(A)

# ── 3. Create Parallel RHS (b) and Solution (x) Vectors ─────────────
b = LibPETSc.VecCreate(petsclib, comm)
LibPETSc.VecSetSizes(petsclib, b, PetscInt(LibPETSc.PETSC_DECIDE), PetscInt(N))
LibPETSc.VecSetFromOptions(petsclib, b)
LibPETSc.VecSetUp(petsclib, b)

for row in rstart:(rend - 1)
    LibPETSc.VecSetValues(petsclib, b, PetscInt(1), [PetscInt(row)], PetscScalar[1.0], LibPETSc.INSERT_VALUES)
end
PETSc.assemble!(b)


# ── 4. Setup KSP Parallel Solver & Solve ─────────────────────────────
ksp = PETSc.KSP(A; ksp_type="bcgs", pc_type="bjacobi", ksp_monitor=true, ksp_view=true)
# ksp = LibPETSc.KSPCreate(petsclib, comm) # ; ksp_type = "cg", pc_type = "jacobi", ksp_monitor = true)
# LibPETSc.KSPSetOperators(petsclib, ksp, A, A)
# LibPETSc.KSPSetFromOptions(petsclib, ksp)
# LibPETSc.KSPSetUp(petsclib, ksp)

x = ksp\b

# ── Cleanup ──────────────────────────────────────────────────────────
PETSc.destroy(ksp)
PETSc.destroy(A)
PETSc.destroy(b)
PETSc.destroy(x)

PETSc.finalize(petsclib)
