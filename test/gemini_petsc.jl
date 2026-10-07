using SparseArrays
using MPI
using PETSc

function getrange(N::Integer, myrank::Integer, commsize::Integer)
    q, r = divrem(N, commsize)
    start_idx = (myrank - 1) * q + min(myrank - 1, r) + 1
    len = q + (myrank <= r ? 1 : 0)
    return start_idx:(start_idx + len - 1)
end

"""
    Laplace_2D_9P_slice(n::Int, row_range::UnitRange{Int}; stencil_type=:standard)

Generates a local `SparseMatrixCSC` slice of size `(length(row_range) × n^2)` 
for a 2D 9-point finite difference Laplacian on an n × n grid.

`stencil_type`:
- `:isotropic` (default): [ -1 -4 -1 ; -4 20 -4 ; -1 -4 -1 ]
- `:simple`:    [ -1 -1 -1 ; -1  8 -1 ; -1 -1 -1 ]
"""
function Laplace_2D_9P_slice(n::Int, row_range::UnitRange{Int}; stencil_type::Symbol = :isotropic)
    N_global = n^2
    N_local  = length(row_range)
    
    # Select stencil weights
    if stencil_type == :isotropic
        val_center = 20.0
        val_orth   = -4.0
        val_diag   = -1.0
    elseif stencil_type == :simple
        val_center = 8.0
        val_orth   = -1.0
        val_diag   = -1.0
    else
        error("Unknown stencil_type: $stencil_type. Use :isotropic or :simple.")
    end

    # Pre-allocate COO vectors (up to 9 entries per row)
    nnz_est = 9 * N_local
    I = sizehint!(Int[], nnz_est)      # Local row index (1 .. N_local)
    J = sizehint!(Int[], nnz_est)      # Global column index (1 .. N_global)
    V = sizehint!(Float64[], nnz_est)  # Non-zero stencil entries

    for (i_loc, i_glob) in enumerate(row_range)
        # Convert 1D global index to 2D grid coordinates (1-based)
        r = div(i_glob - 1, n) + 1
        c = mod(i_glob - 1, n) + 1

        # Center (diagonal)
        push!(I, i_loc); push!(J, i_glob); push!(V, val_center)

        # Orthogonal neighbors (North, South, East, West)
        if c > 1 ; push!(I, i_loc); push!(J, i_glob - 1); push!(V, val_orth); end # West
        if c < n ; push!(I, i_loc); push!(J, i_glob + 1); push!(V, val_orth); end # East
        if r > 1 ; push!(I, i_loc); push!(J, i_glob - n); push!(V, val_orth); end # South
        if r < n ; push!(I, i_loc); push!(J, i_glob + n); push!(V, val_orth); end # North

        # Diagonal neighbors (SW, SE, NW, NE)
        if r > 1 && c > 1 ; push!(I, i_loc); push!(J, i_glob - n - 1); push!(V, val_diag); end # South-West
        if r > 1 && c < n ; push!(I, i_loc); push!(J, i_glob - n + 1); push!(V, val_diag); end # South-East
        if r < n && c > 1 ; push!(I, i_loc); push!(J, i_glob + n - 1); push!(V, val_diag); end # North-West
        if r < n && c < n ; push!(I, i_loc); push!(J, i_glob + n + 1); push!(V, val_diag); end # North-East
    end

    return (rowval=I, colptr=J, nzval=V, m=N_local, n=N_global)
end

function Laplace_2D_5P_slice(n::Int, row_range::UnitRange{Int})
    N_global = n^2
    N_local  = length(row_range)
    
    # Pre-allocate COO arrays (approx. 5 non-zeros per row)
    nnz_est = 5 * N_local
    I = sizehint!(Int[], nnz_est)      # Local row indices: 1 .. N_local
    J = sizehint!(Int[], nnz_est)      # Global column indices: 1 .. N_global
    V = sizehint!(Float64[], nnz_est)  # Stencil values
    
    for (i_loc, i_glob) in enumerate(row_range)
        # Convert global row index to 2D grid coordinates (1-based)
        r = div(i_glob - 1, n) + 1
        c = mod(i_glob - 1, n) + 1
        
        # Center (diagonal)
        push!(I, i_loc)
        push!(J, i_glob)
        push!(V, 4.0)
        
        # Left neighbor
        if c > 1
            push!(I, i_loc)
            push!(J, i_glob - 1)
            push!(V, -1.0)
        end
        
        # Right neighbor
        if c < n
            push!(I, i_loc)
            push!(J, i_glob + 1)
            push!(V, -1.0)
        end
        
        # Bottom neighbor
        if r > 1
            push!(I, i_loc)
            push!(J, i_glob - n)
            push!(V, -1.0)
        end
        
        # Top neighbor
        if r < n
            push!(I, i_loc)
            push!(J, i_glob + n)
            push!(V, -1.0)
        end
    end
    
    return (rowval=I, colptr=J, nzval=V, m=N_local, n=N_global)
end

if !MPI.Initialized()
    MPI.Init()
end

args = [
    "-ksp_type", "bcgsl",
    "-ksp_bcgsl_ell", "2",
    "-pc_type", "asm",
    "-pc_asm_overlap", "0",          # Overlap width (2 grid point)
    "-pc_asm_type", "restrict",       # "restrict" = RAS, "basic" = Standard ASM
     "-sub_pc_type", "ilu",            # Local preconditioner on each rank
     "-sub_pc_factor_levels", "4",     # ILU(0)
    "-ksp_rtol", "1e-8",
    "-ksp_view"
]

comm = MPI.COMM_WORLD
petsclib = first(PETSc.petsclibs)
PETSc.initialized(petsclib) || PETSc.initialize(petsclib; options=args)

PetscScalar = PETSc.scalartype(petsclib)
PetscInt    = PETSc.inttype(petsclib)

# ── Problem Setup (2D Poisson grid: n x n) ─────────────────────────
n = 1000
N = n^2 # Total global system size

# ── 1. Create MPI Parallel Matrix (A) ────────────────────────────────
A = LibPETSc.MatCreate(petsclib, comm)
LibPETSc.MatSetSizes(petsclib, A, PetscInt(LibPETSc.PETSC_DECIDE), PetscInt(LibPETSc.PETSC_DECIDE), PetscInt(N), PetscInt(N))
# LibPETSc.MatSetType(petsclib, A, "mpiaij") # Automatically picks MATMPIAIJ for size(comm) > 1
LibPETSc.MatSetUp(petsclib, A)
rstart, rend = LibPETSc.MatGetOwnershipRange(petsclib, A)

myrange = getrange(N, MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))
S = Laplace_2D_9P_slice(n, myrange)
nelements = length(S.nzval)
LibPETSc.MatSetPreallocationCOO(petsclib, A, PetscInt(nelements), PetscInt.(S.rowval .-1 .+ rstart), PetscInt.(S.colptr .-1))
LibPETSc.MatSetValuesCOO(petsclib, A, PetscScalar.(S.nzval), LibPETSc.INSERT_VALUES)

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
ksp = PETSc.KSP(A; ksp_monitor=true, ksp_view=true)
# pc = Ref{LibPETSc.PC}()
# LibPETSc.KSPGetPC(petsclib, ksp, pc)
# LibPETSc.PCASMSetOverlap(petsclib, pc, PetscInt(1))
# 
# LibPETSc.KSPSetupFromOptions(petsclib, ksp)
# LibPETSc.KSPSetUp(petsclib, ksp)

# ksp = LibPETSc.KSPCreate(petsclib, comm) # ; ksp_type = "cg", pc_type = "jacobi", ksp_monitor = true)
# LibPETSc.KSPSetOperators(petsclib, ksp, A, A)
LibPETSc.KSPSetFromOptions(petsclib, ksp)
LibPETSc.KSPSetUp(petsclib, ksp)

x = ksp\b

# ── Cleanup ──────────────────────────────────────────────────────────
PETSc.destroy(ksp)
PETSc.destroy(A)
PETSc.destroy(b)
PETSc.destroy(x)

PETSc.finalize(petsclib)
