module ParallelSolver

using IterativeSolvers
using IncompleteLU
using MPI
using LinearAlgebra
using SparseArrays
import IncompleteLU: ILUFactorization

export DistributedMatrix
export DistributedArray, DistributedVector
export RASPreconditioner

include("distributedArray.jl")
include("distributedMatrix.jl")
include("schwarzDecomposition.jl")

end
