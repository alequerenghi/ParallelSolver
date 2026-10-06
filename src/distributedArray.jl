struct DistributedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    loc::A
    comm::MPI.Comm
end

const DistributedVector{T,A} = DistributedArray{T,1,A}

# DistributedArray(loc::AbstractArray{T,N}, comm::MPI.Comm) where {T,N} =
#     DistributedArray{T,N,typeof(loc)}(loc, comm)
DistributedVector(loc::AbstractVector{T}, comm::MPI.Comm) where {T} =
    DistributedArray(loc, comm)
DistributedVector{T}(loc::AbstractVector{T}, comm::MPI.Comm) where {T} =
    DistributedArray(loc, comm)

Base.size(x::DistributedArray) = size(x.loc)
Base.getindex(x::DistributedArray, I::Vararg{Int,N}) where {N} =
    getindex(x.loc, I...)
Base.setindex!(x::DistributedArray, v, I::Vararg{Int,N}) where {N} =
    setindex!(x.loc, v, I...)
Base.IndexStyle(::Type{<:DistributedArray{T,N,A}}) where {T,N,A} = IndexStyle(A)
Base.similar(A::DistributedArray, ::Type{S}, dims::Dims) where {S} =
    DistributedArray(similar(A.loc, S, dims), A.comm)

@inline local_slice(v::AbstractArray) = v
@inline local_slice(v::DistributedArray) = v.loc
# @inline local_slice(v::SubArray{T,N,<:Vector}) where {T,N} = v
@inline local_slice(v::SubArray{T,N,<:DistributedArray}) where {T,N} =
    view(v.parent.loc, parentindices(v)...)

@inline get_comm(v::DistributedArray) = v.comm
@inline get_comm(v::SubArray{T,N,<:DistributedArray}) where {T,N} =
    parent(v).comm

function LinearAlgebra.dot(
    x::Union{DistributedArray{T,N},SubArray{T,N,<:DistributedArray}},
    y::Union{DistributedArray{T,N},SubArray{T,N,<:DistributedArray}},
) where {T,N}
    local_dot = dot(local_slice(x), local_slice(y))
    return MPI.Allreduce(local_dot, MPI.SUM, get_comm(x))
end

function LinearAlgebra.norm(
    x::Union{DistributedArray{T,N},SubArray{T,N,<:DistributedArray}},
) where {T,N}
    return sqrt(dot(x, x))
end

# Gram matrix C = A' * B, reduced over all ranks (used by the MR step of bicgstabl)
function LinearAlgebra.mul!(
    C::Union{Matrix{T},DistributedArray{T,2}},
    At::Adjoint{T,<:DistributedArray{T,2}},
    B::DistributedArray{T,2},
) where {T}
    Cloc = local_slice(C)
    mul!(Cloc, adjoint(parent(At).loc), B.loc)
    MPI.Allreduce!(Cloc, MPI.SUM, B.comm)
    return C
end

