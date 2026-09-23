using Printf
using IterativeSolvers
using IncompleteLU
using MPI
using LinearAlgebra
using SparseArrays

struct DistributedArray{T,N,A <: AbstractArray{T,N}} <: AbstractArray{T,N}
    loc::A
    comm::MPI.Comm
end

const DistributedVector{T,A} = DistributedArray{T,1,A}

DistributedArray(loc::AbstractArray{T,N}, comm::MPI.Comm) where {T,N} = DistributedArray{T,N,typeof(loc)}(loc, comm)
DistributedVector(loc::AbstractVector{T}, comm::MPI.Comm) where {T} = DistributedArray(loc, comm)
DistributedVector{T}(loc::AbstractVector{T}, comm::MPI.Comm) where {T} = DistributedArray(loc, comm)

Base.size(x::DistributedArray) = size(x.loc)
Base.getindex(x::DistributedArray, I::Vararg{Int, N}) where {N} = getindex(x.loc, I...)
Base.setindex!(x::DistributedArray, v, I::Vararg{Int, N}) where {N} = setindex!(x.loc, v, I...)
Base.IndexStyle(::Type{<:DistributedArray{T, N, A}}) where {T, N, A} = IndexStyle(A)
Base.similar(A::DistributedArray, ::Type{S}, dims::Dims) where {S} = 
    DistributedArray(similar(A.loc, S, dims), A.comm)

@inline local_slice(v::AbstractArray) = v 
@inline local_slice(v::DistributedArray) = v.loc
# @inline local_slice(v::SubArray{T,N,<:Vector}) where {T,N} = v
@inline local_slice(v::SubArray{T,N,<:DistributedArray}) where {T,N} = view(v.parent.loc, parentindices(v)...)

@inline get_comm(v::DistributedArray) = v.comm
@inline get_comm(v::SubArray{T,N,<:DistributedArray}) where {T,N} = parent(v).comm

function LinearAlgebra.dot(
    x::Union{DistributedArray{T, N}, SubArray{T, N, <:DistributedArray}}, 
    y::Union{DistributedArray{T, N}, SubArray{T, N, <:DistributedArray}}
) where {T, N}
    local_dot = dot(local_slice(x), local_slice(y))
    return MPI.Allreduce(local_dot, MPI.SUM, get_comm(x))
end

function LinearAlgebra.norm(
    x::Union{DistributedArray{T, N}, SubArray{T, N, <:DistributedArray}}
) where {T, N}
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
    ghostmap  = Dict{Ti, Ti}()
    @inbounds for j ∈ 1:A.n
        if   startᵢ <= j <= stopᵢ
            jloc = j - startᵢ + 1
            for k ∈ nzrange(A, j)
                push!(Id, A.rowval[k])
                push!(Jd,  jloc)
                push!(Vd, A.nzval[k])
            end
        elseif !isempty(nzrange(A, j))
            ghostᵢ += 1
            ghostmap[j] = ghostᵢ
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
    return Mₗ, Mₒ, ghostmap
end

struct DistributedMatrix{Tv,Ti}
    loc::SparseMatrixCSC{Tv,Ti}
    int::SparseMatrixCSC{Tv,Ti}
    ghosts::Vector{Tv}
    recv_sizes::Vector{Ti}
    sendmap::Vector{Vector{Ti}}
    recvmap::Vector{Vector{Ti}}
    sendbufs::Vector{Vector{Tv}}
    recvbufs::Vector{Vector{Tv}}
    cs::Vector{Ti}
    ghostmap::Dict{Ti,Ti}
    ghost_global::Vector{Ti}
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

        Mlocal, Mghosts, ghostmap = separate_local_ghosts(A, cs, ng, myrank)

        ghosts    = Vector{Tv}(undef, size(Mghosts, 2))
        sendbufs  = [Vector{Tv}(undef, length(sendmap[i])) for i = 1:commsize]
        recv_sizes  = cumsum([1; length.(recvmap)])
        ghost_global = Vector{Ti}(undef, length(ghostmap))
        for (k, v) ∈ ghostmap
            ghost_global[v] = k
        end

        reqcount  = count(!isempty, recvmap) + count(!isempty, sendmap)
        recvbufs  = [view(ghosts, recv_sizes[r]:recv_sizes[r+1]-1) for r in 1:commsize]

        return new{Tv,Ti}(Mlocal, Mghosts, ghosts, recv_sizes, sendmap, recvmap, sendbufs, recvbufs, cs, ghostmap, ghost_global, reqcount, comm)
    end
end

function ghostexchange!(
        ghosts::Vector{AbstractArray{Tv}},
        source::Vector{Tv},
        sendbufs::Vector{Vector{Tv}},
        sendmap::Vector{Vector{Integer}},
        recv_sizes::Vector{Integer},
        reqcount::Integer,
        comm::MPI.Comm
) where {Tv}
    commsize  = MPI.Comm_size(comm)

    reqs = MPI.MultiRequest(reqcount)
    req = 1
    for rank in 1:commsize
        buf = recvbufs[rank]
        if !isempty(buf)
            MPI.Irecv!(buf, comm, reqs[req]; source=rank-1)
            req += 1
        end
    end
    for rank ∈ 1:commsize
        if !isempty(sendbufs[rank])
            map!(i -> source[i], sendbufs[rank], sendmap[rank])
            MPI.Isend(sendbufs[rank], comm, reqs[req]; dest=rank-1)
            req += 1
        end
    end
    return reqs
end

function ghostexchange!(A::DistributedMatrix{Tv,Ti}, x::AbstractVector{Tv}) where {Tv,Ti} 
    ghostexchange!{Tv}(A.recvbufs, x, A.sendbufs, A.sendmap, A.recv_sizes, A.reqcount, A.comm)
    for rank ∈ 1:commsize
        ids = A.recvmap[rank]
        data = A.recvbufs[rank]
        for j ∈ eachindex(ids)
            jglobal = ids[j]
            A.buf[A.ghostmap[j]] = data[j]
        end
    end
    return nothing
end

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
Base.size(M::DistributedMatrix, d::Integer) = d == 1 ? M.loc.m : (2 == d ? M.loc.n : 1)

const BufferType{Tv,Ti} = @NamedTuple{
    i::Vector{Vector{Ti}},
    j::Vector{Vector{Ti}},
    v::Vector{Vector{Tv}}
}

function addrows!(send_counts, outbufs::BufferType{Tv,Ti}, offset, loc, interface, sendmap, ghostglobal) where {Tv,Ti}
    commsize = length(outbufs.i)
    for rank ∈ 1:commsize
        for ilocal ∈ sendmap[rank]
            nzcount = 0
            for k ∈ nzrange(loc, ilocal)
                nzcount += 1
                j = loc.rowval[k]
                v = loc.nzval[k]
                push!(outbufs.j[rank], j + offset)
                push!(outbufs.v[rank], v)
            end
            for k ∈ nzrange(interface, ilocal)
                nzcount += 1
                j = interface.rowval[k]
                v = interface.nzval[k]
                push!(outbufs.j[rank], ghostglobal[j])
                push!(outbufs.v[rank], v)
            end
            push!(outbufs.i[rank], nzcount)
            nnz_loc = length(nzrange(loc, ilocal))
            nnz_int = length(nzrange(interface, ilocal))
            send_counts[rank] += nnz_loc + nnz_int
        end
    end
end

function exchangerows!(inbufs::BufferType{Tv,Ti}, outbufs::BufferType{Tv,Ti}, comm::MPI.Comm) where {Tv,Ti}
    commsize  = length(inbufs.i)
    reqs = MPI.Request[]
    for rank ∈ 1:commsize
        if !isempty(inbufs.i[rank])
            push!(reqs, MPI.Irecv!(inbufs.i[rank], comm; source=rank-1, tag=0))
        end
        if !isempty(inbufs.j[rank])
            push!(reqs, MPI.Irecv!(inbufs.j[rank], comm; source=rank-1, tag=1))
            push!(reqs, MPI.Irecv!(inbufs.v[rank], comm; source=rank-1, tag=2))
        end
    end
    for rank ∈ 1:commsize
        if !isempty(outbufs.i[rank])
            push!(reqs, MPI.Isend(outbufs.i[rank], comm; dest=rank-1, tag=0))
        end
        if !isempty(outbufs.j[rank])
            push!(reqs, MPI.Isend(outbufs.j[rank], comm; dest=rank-1, tag=1))
            push!(reqs, MPI.Isend(outbufs.v[rank], comm; dest=rank-1, tag=2))
        end
    end
    MPI.Waitall(reqs)
    return nothing
end

function appendghostrows!(data, inbufs::BufferType{Tv,Ti}, ghostmap, recvmaps, cs, N_local) where {Tv,Ti}
    commsize = length(inbufs.i)
    offset    = -cs[myrank]+1
    for rank ∈ 1:commsize
        is          = inbufs.i[rank]
        js          = inbufs.j[rank]
        vs          = inbufs.v[rank]
        recvs       = recvmaps[rank]
        cum_icount  = cumsum([1; is])
        for i ∈ 1:length(is)
            ilocal = ghostmap[recvs[i]]
            for k ∈ cum_icount[i]:(cum_icount[i+1]-1)
                jlocal = js[k]
                if cs[myrank] <= jlocal <= cs[myrank+1]-1
                    push!(data.j, jlocal + offset)
                elseif jlocal ∈ keys(ghostmap)
                    push!(data.j, N_local + ghostmap[jlocal])
                else
                    continue
                end
                push!(data.i, N_local + ilocal)
                push!(data.v, vs[k])
            end
        end
    end
    return nothing
end

function schwarz!(sendmaps::Vector{Vector{Ti}}, recvmaps::Vector{Vector{Ti}}, ghostmap::Dict{Ti,Ti}, M::DistributedMatrix{Tv,Ti}, overlap=1) where {Tv,Ti}
    comm = M.comm
    commsize  = MPI.Comm_size(comm)
    myrank    = MPI.Comm_rank(comm)+1
    N_local   = size(M.loc, 1)
    nghosts   = size(M.int, 2)
    N_overlap = N_local + nghosts

    I, J, V     = findnz(M.loc)
    Io, Jo, Vo  = findnz(M.int)
    append!(I, Io)
    append!(J, Jo .+= N_local)
    append!(V, Vo)

    A = BufferType{Tv,Ti}((I, J, V))

    loc_T = sparse(M.loc')
    int_T = sparse(M.int')
    outbufs = BufferType{Tv,Ti}((
        [Ti[] for _ ∈ 1:commsize],
        [Ti[] for _ ∈ 1:commsize],
        [Tv[] for _ ∈ 1:commsize],
       ))
    offlocal = M.cs[myrank]-1
    send_counts = zeros(Ti, commsize)
    addrows!(
        send_counts,
        outbufs,
        offlocal,
        loc_T,
        int_T,
        M.sendmap,
        M.ghost_global
    )

    recv_counts = MPI.Alltoall(UBuffer(send_counts,1), comm)
    inbufs = BufferType{Tv,Ti}((
        [Vector{Ti}(undef, length(M.recvmap[r])) for r ∈ 1:commsize],
        [Vector{Ti}(undef, nj) for nj in recv_counts],
        [Vector{Tv}(undef, nv) for nv in recv_counts],
       ))

    exchangerows!(inbufs, outbufs, comm)

    appendghostrows!(
        A,
        inbufs,
        M.ghostmap,
        M.recvmap,
        M.cs,
        N_local
    )

    sparse(I, J, V, N_overlap, N_overlap)
end

struct RASPreconditioner{T, F}
    Pl::F
    N_local::Int
    buf::Vector{T}
    recv_sizes::Vector{Ti}
    sendmap::Vector{Vector{Ti}}
    recvmap::Vector{Vector{Ti}}
    sendbufs::Vector{Vector{Tv}}
    recvbufs::Vector{Vector{Tv}}
    ghostmap::Dict{Ti,Ti}
    reqcount::Int
    comm::MPI.Comm
    function RASPreconditioner(M::DistributedMatrix{Tv,Ti}, overlap=2; τ=1e-3) where {Tv,Ti}
        S   = schwarz!(M, overlap)
        Pl = ilu(M; τ=τ)
        buf = Vector{Tv}(undef, size(S,1))
        return new{ILUFactorization, Tv}(S, size(M.loc, 1), buf)
    end
end
function ghostexchange!(P::RASPreconditioner, x::Vector{Tv}) where {Tv} 
    ghostexchange!{Tv}(P.recvbufs, x, P.sendbufs, P.sendmap, P.recv_sizes, P.reqcount, P.comm)
    for rank ∈ 1:commsize
        ids = P.recvmap[rank]
        data = P.recvbufs[rank]
        for j ∈ eachindex(ids)
            jglobal = ids[j]
            P.buf[P.ghostmap[j]] = data[j]
        end
    end
    return nothing
end

function LinearAlgebra.ldiv!(A::RASPreconditioner, b)
    copyto!(A.buf, 1, b, 1, length(b))

    ghostexchange!(A)

    ldiv!(A.Pl, A.buf)

    copyto(b, 1, A.buf, 1, length(b))
end

function IterativeSolvers.bicgstabl_iterator!(x, A::DistributedMatrix, b, l::Int = 2;
                             Pl = Identity(),
                             max_mv_products = size(A, 2),
                             abstol::Real = zero(real(eltype(b))),
                             reltol::Real = sqrt(eps(real(eltype(b)))),
                             initial_zero = false)
    comm = A.comm
    T = eltype(x)
    n = size(A, 1)
    mv_products = 0

    # Large vectors.
    # Should become distributed
    r_shadow = DistributedVector(rand(T, n), comm)
    # Also must become distributed
    rs = DistributedArray(Matrix{T}(undef, n, l + 1), comm)
    us = DistributedArray(zeros(T, n, l + 1), comm)

    residual = view(rs, :, 1)

    # Compute the initial residual rs[:, 1] = b - A * x
    # Avoid computing A * 0.
    if initial_zero
        copyto!(residual, b)
    else
        mul!(residual, A, x)
        residual .= b .- residual
        mv_products += 1
    end

    # Apply the left preconditioner
    ldiv!(Pl, residual)

    γ = DistributedVector(zeros(T, l), comm)
    ω = σ = one(T)

    nrm = norm(residual)

    # For the least-squares problem
    M = DistributedArray(zeros(T, l + 1, l + 1), comm)

    # Stopping condition based on absolute and relative tolerance.
    tolerance = max(reltol * nrm, abstol)

    BiCGStabIterable(A, l, x, r_shadow, rs, us,
        max_mv_products, mv_products, tolerance, nrm,
        Pl,
        γ, ω, σ, M
    )
end
