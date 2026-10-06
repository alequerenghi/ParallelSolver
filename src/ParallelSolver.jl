module ParallelSolver

using Printf
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
using ParallelSolver
include("schwarzDecomposition.jl")

end
