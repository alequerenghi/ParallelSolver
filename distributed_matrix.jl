using IncompleteLU
using MPI
using LinearAlgebra
using SparseArrays

struct DistributedVector{T} <: AbstractVector{T}
    loc::Vector{T}
    comm::MPI.Comm
end

Base.size(x::DistributedVector) = size(x.loc)
Base.getindex(x::DistributedVector, i::Int) = x.loc[i]
Base.setindex!(x::DistributedVector, v, i::Int) = (x.loc[i] = v)
Base.similar(x::DistributedVector, ::Type{S}) where {S} = DistributedVector(similar(x.loc, S), x.comm)

function LinearAlgebra.dot(x::DistributedVector{T}, y::DistributedVector{T}) where {T}
    local_dot = dot(x.loc, y.loc)
    return MPI.Allreduce(local_dot, MPI.SUM, x.comm)
end

function LinearAlgebra.norm(x::DistributedVector{T}) where {T}
    return sqrt(dot(x, x))
end



function scanghosts!(
    recvmap::Vector{Vector{Ti}},
    A::SparseMatrixCSC{Tv,Ti},
    cs::Vector{Int},
    myrank::Int,
) where {Tv,Ti}
    ng     = 0
    ghosts = Ti[]
    startᵢ = cs[myrank]
    stopᵢ  = cs[myrank+1] - 1
    @inbounds begin
        for j = 1:A.n
            ngj = length(nzrange(A, j))
            if (j < startᵢ || j > stopᵢ) && 0 < ngj
                push!(ghosts, j)
                ng += ngj
            end
        end
        if !isempty(ghosts)
            firstneighbor = searchsortedlast(cs, ghosts[1])
            lastneighbor = searchsortedlast(cs, ghosts[end])
            ghostsperrank = ceil(Ti, length(ghosts) / (lastneighbor - firstneighbor + 1))
            for r ∈ firstneighbor:lastneighbor
                sizehint!(recvmap[r], ghostsperrank)
            end

            for g ∈ ghosts
                r = searchsortedlast(cs, g)
                push!(recvmap[r], g)
            end
        end
    end
    return ng
end

function build_sendmap(
    recvmap::Vector{Vector{Ti}},
    cs::Vector{Int},
    comm::MPI.Comm,
) where {Ti}
    commsize = MPI.Comm_size(comm)
    myrank   = MPI.Comm_rank(comm)+1
    recv_counts = length.(recvmap)
    send_counts = MPI.Alltoall(UBuffer(recv_counts, 1), comm)
    reqs = MPI.Request[]
    sendmap = Vector{Vector{Ti}}(undef, commsize)
    for (rank, counts) ∈ enumerate(send_counts)
        buf = Vector{Ti}(undef, counts)
        sendmap[rank] = buf
        if 0 < counts
            req = MPI.Irecv!(buf, comm; source = rank-1)
            push!(reqs, req)
        end
    end
    for (rank, v) ∈ enumerate(recvmap)
        if !isempty(v)
            req = MPI.Isend(v, comm; dest = rank-1)
            push!(reqs, req)
        end
    end
    MPI.Waitall(reqs)
    local_off = cs[myrank] - 1
    for v ∈ sendmap
        if !isempty(v)
            v .-= local_off
        end
    end
    return sendmap
end

function separate_local_ghosts(
    A::SparseMatrixCSC{Tv,Ti},
    cs::Vector{Int},
    nghosts::Int,
    myrank::Int,
) where {Tv,Ti}
    startᵢ = cs[myrank]
    stopᵢ  = cs[myrank+1] - 1
    local_count = nnz(A) - nghosts

    Id = sizehint!(Ti[], local_count)
    Jd = sizehint!(Ti[], local_count)
    Vd = sizehint!(Tv[], local_count)

    Io = sizehint!(Ti[], nghosts)
    Jo = sizehint!(Ti[], nghosts)
    Vo = sizehint!(Tv[], nghosts)

    ghostᵢ = 0
    @inbounds for j ∈ 1:A.n
        if   startᵢ <= j <= stopᵢ
            jloc = j - startᵢ + 1
            for k ∈ nzrange(A, j)
                push!(Id, A.rowval[k])
                push!(Jd,  jloc)
                push!(Vd, A.nzval[k])
            end
        else
            ghostᵢ += 1
            for k ∈ nzrange(A, j)
                push!(Io, A.rowval[k])
                push!(Jo, ghostᵢ)
                push!(Vo, A.nzval[k])
            end
        end
    end
    N_local = stopᵢ - startᵢ + 1

    Mₗ = sparse(Id, Jd, Vd, N_local, N_local)
    Mₒ = sparse(Io, Jo, Vo, N_local, ghostᵢ)
    return Mₗ, Mₒ
end

struct DistributedMatrix{Tv,Ti}
    loc::SparseMatrixCSC{Tv,Ti}
    int::SparseMatrixCSC{Tv,Ti}
    ghosts::Vector{Tv}
    recv_sizes::Vector{Ti}
    sendmap::Vector{Vector{Ti}}
    sendbufs::Vector{Vector{Tv}}
    reqcount::Int
    comm::MPI.Comm
    function DistributedMatrix(
        A::SparseMatrixCSC{Tv,Ti},
        comm::MPI.Comm,
    ) where {Tv,Ti}
        commsize  = MPI.Comm_size(comm)
        myrank    = MPI.Comm_rank(comm) + 1
        m         = size(A, 1)

        sizes = MPI.Allgather(m, comm)
        cs    = cumsum([1; sizes])

        recvmap = [Ti[] for _ ∈ 1:commsize]
        ng      = scanghosts!(recvmap, A, cs, myrank)
        sendmap = build_sendmap(recvmap, cs, comm)

        Mlocal, Mghosts = separate_local_ghosts(A, cs, ng, myrank)

        ghosts    = Vector{Tv}(undef, size(Mghosts, 2))
        sendbufs  = [Vector{Tv}(undef, length(sendmap[i])) for i = 1:commsize]
        recv_sizes  = cumsum([1; length.(recvmap)])
        reqcount  = count(!isempty, recvmap) + count(!isempty, sendmap)

        return new{Tv,Ti}(Mlocal, Mghosts, ghosts, recv_sizes, sendmap, sendbufs, reqcount, comm)
    end
end

function ghostexchange!(A::DistributedMatrix{Tv,Ti}, x::AbstractVector{Tv}) where {Tv,Ti}
    comm      = A.comm
    commsize  = MPI.Comm_size(comm)

    reqs = MPI.MultiRequest(A.reqcount)
    req = 1
    for rank in 1:commsize
        datarange = A.recv_sizes[rank]:A.recv_sizes[rank+1]-1
        if 0 < length(datarange)
            buf = view(A.ghosts, datarange)
            MPI.Irecv!(buf, comm, reqs[req]; source=rank-1)
            req += 1
        end
    end
    for rank ∈ 1:commsize
        if !isempty(A.sendbufs[rank])
            map!(i -> x[i], A.sendbufs[rank], A.sendmap[rank])
            MPI.Isend(A.sendbufs[rank], comm, reqs[req]; dest=rank-1)
            req += 1
        end
    end
    return reqs
end

@inline local_slice(v::AbstractVector) = v 
@inline local_slice(v::SubArray{T,N,<:Vector}) where {T,N} = v
@inline local_slice(v::DistributedVector) = v.loc
@inline local_slice(v::SubArray{T,N,<:DistributedVector}) where {T,N} = view(v.parent.loc, parentindices(v))

function LinearAlgebra.mul!(y::AbstractVector{Tv}, M::DistributedMatrix{Tv,Ti}, x::AbstractVector{Tv}) where {Tv,Ti}
    yloc  = local_slice(y)
    xloc  = local_slice(x)
    reqs  = ghostexchange!(M, xloc)
    mul!(yloc, M.loc, xloc)
    MPI.Waitall(reqs)
    mul!(yloc, M.int, M.ghosts, one(Tv), one(Tv))
    return y
end

function LinearAlgebra.mul!(y::AbstractVector{Tv}, M::DistributedMatrix{Tv,Ti}, x::AbstractVector{Tv}, α::Number, β::Number) where {Tv,Ti}
    yloc  = local_slice(y)
    xloc  = local_slice(x)
    reqs  = ghostexchange!(M, xloc)
    mul!(yloc, M.loc, xloc, α, β)
    MPI.Waitall(reqs)
    mul!(yloc, M.int, M.ghosts, α, one(Tv))
    return y
end

IncompleteLU.ilu(M::DistributedMatrix{Tv,Ti}; τ=1e-3) where {Tv,Ti} = ilu(M.loc; τ)

Base.size(M::DistributedMatrix) = (M.loc.m, M.loc.n)
Base.size(M::DistributedMatrix, d::Integer) = d == 1 ? M.loc.m : (2 == d ? A.loc.n : 1)

