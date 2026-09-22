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

        return new{Tv,Ti}(Mlocal, Mghosts, ghosts, recv_sizes, sendmap, recvmap, sendbufs, cs, ghostmap, ghost_global, reqcount, comm)
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

function schwarz!(M::DistributedMatrix{Tv,Ti}; overlap=1) where {Tv,Ti}
    comm = M.comm
    commsize  = MPI.Comm_size(comm)
    myrank    = MPI.Comm_rank(comm)+1
    N_local = size(M.loc, 1)
    nghosts = size(M.int, 2)
    N_overlap = N_local + nghosts

    I, J, V = findnz(M.loc)
    Io, Jo, Vo = findnz(M.int)
    append!(I, Io)
    Jo .+= N_local
    append!(J, Jo)
    append!(V, Vo)

    loc_T = sparse(M.loc')
    int_T = sparse(M.int')
    iout  = [Ti[] for _ ∈ 1:commsize]
    jout  = [Ti[] for _ ∈ 1:commsize]
    vout  = [Tv[] for _ ∈ 1:commsize]
    offlocal  = M.cs[myrank]-1
    send_counts = zeros(Ti, commsize)
    for rank ∈ 1:commsize
        for ilocal ∈ M.sendmap[rank]
            nzcount = 0
            for k ∈ nzrange(loc_T, ilocal)
                nzcount += 1
                j = loc_T.rowval[k]
                v = loc_T.nzval[k]
                push!(jout[rank], j + offlocal)
                push!(vout[rank], v)
            end
            for k ∈ nzrange(int_T, ilocal)
                nzcount += 1
                j = int_T.rowval[k]
                v = int_T.nzval[k]
                push!(jout[rank], M.ghost_global[j])
                push!(vout[rank], v)
            end
            push!(iout[rank], nzcount)
            nnz_loc = length(nzrange(loc_T, ilocal))
            nnz_int = length(nzrange(int_T, ilocal))
            send_counts[rank] += nnz_loc + nnz_int
        end
    end
    recv_counts = MPI.Alltoall(UBuffer(send_counts,1), comm)
    iin   = [Vector{Ti}(undef, length(M.recvmap[r])) for r ∈ 1:commsize]
    jin   = [Vector{Ti}(undef, nj) for nj in recv_counts]
    vin   = [Vector{Tv}(undef, nv) for nv in recv_counts]
    reqs = MPI.Request[]
    for rank ∈ 1:commsize
        if !isempty(iin[rank])
            push!(reqs, MPI.Irecv!(iin[rank], comm; source=rank-1, tag=0))
        end
        if !isempty(jin[rank])
            push!(reqs, MPI.Irecv!(jin[rank], comm; source=rank-1, tag=1))
            push!(reqs, MPI.Irecv!(vin[rank], comm; source=rank-1, tag=2))
        end
    end
    for rank ∈ 1:commsize
        if !isempty(iout[rank])
            push!(reqs, MPI.Isend(iout[rank], comm; dest=rank-1, tag=0))
        end
        if !isempty(jout[rank])
            push!(reqs, MPI.Isend(jout[rank], comm; dest=rank-1, tag=1))
            push!(reqs, MPI.Isend(vout[rank], comm; dest=rank-1, tag=2))
        end
    end
    MPI.Waitall(reqs)
    for rank ∈ 1:commsize
        cum_icount = cumsum([1; iin[rank]])
        for i ∈ 1:length(iin[rank])
            ilocal = M.ghostmap[M.recvmap[rank][i]]
            for k ∈ cum_icount[i]:cum_icount[i+1]-1
                jlocal = jin[rank][k]
                if M.cs[myrank] <= jlocal <= M.cs[myrank+1]-1
                    push!(J, jlocal - M.cs[myrank]+1)
                elseif jlocal ∈ keys(M.ghostmap)
                    push!(J, N_local + M.ghostmap[jlocal])
                else
                    continue
                end
                push!(I, N_local + ilocal)
                push!(V, vin[rank][k])
            end
        end
    end
    sparse(I, J, V, N_overlap, N_overlap)
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
