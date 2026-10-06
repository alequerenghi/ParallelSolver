const COOType{Tv,Ti} = @NamedTuple begin
    i::Vector{Ti}
    j::Vector{Ti}
    v::Vector{Tv}
end

const BufferType{Tv,Ti} = @NamedTuple{
    i::Vector{Vector{Ti}},
    j::Vector{Vector{Ti}},
    v::Vector{Vector{Tv}},
}

function addrows!(
    send_counts,
    outbufs::BufferType{Tv,Ti},
    offset,
    loc,
    interface,
    sendmap,
    ghostglobal,
) where {Tv,Ti}
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

function exchangerows!(
    inbufs::BufferType{Tv,Ti},
    outbufs::BufferType{Tv,Ti},
    comm::MPI.Comm,
) where {Tv,Ti}
    commsize = length(inbufs.i)
    reqs = MPI.Request[]
    for rank ∈ 1:commsize
        if !isempty(inbufs.i[rank])
            push!(
                reqs,
                MPI.Irecv!(inbufs.i[rank], comm; source = rank-1, tag = 0),
            )
        end
        if !isempty(inbufs.j[rank])
            push!(
                reqs,
                MPI.Irecv!(inbufs.j[rank], comm; source = rank-1, tag = 1),
            )
            push!(
                reqs,
                MPI.Irecv!(inbufs.v[rank], comm; source = rank-1, tag = 2),
            )
        end
    end
    for rank ∈ 1:commsize
        if !isempty(outbufs.i[rank])
            push!(
                reqs,
                MPI.Isend(outbufs.i[rank], comm; dest = rank-1, tag = 0),
            )
        end
        if !isempty(outbufs.j[rank])
            push!(
                reqs,
                MPI.Isend(outbufs.j[rank], comm; dest = rank-1, tag = 1),
            )
            push!(
                reqs,
                MPI.Isend(outbufs.v[rank], comm; dest = rank-1, tag = 2),
            )
        end
    end
    MPI.Waitall(reqs)
    return nothing
end

function appendghostrows!(
    recvmap_new,
    data,
    inbufs::BufferType{Tv,Ti},
    ghostmap,
    recvmaps,
    cs,
    start,
    stop,
    N_local,
    remaining,
) where {Tv,Ti}
    commsize = length(inbufs.i)
    offset = -start+1
    newrows = length(ghostmap)
    for rank ∈ 1:commsize
        is         = inbufs.i[rank]
        js         = inbufs.j[rank]
        vs         = inbufs.v[rank]
        recvs      = recvmaps[rank]
        cum_icount = cumsum([1; is])
        for i ∈ 1:length(is)
            ilocal = ghostmap[recvs[i]]
            for k ∈ cum_icount[i]:(cum_icount[i+1]-1)
                jlocal = js[k]
                if start <= jlocal <= stop
                    push!(data.j, jlocal + offset)
                elseif jlocal ∈ keys(ghostmap)
                    push!(data.j, N_local + ghostmap[jlocal])
                elseif 0 < remaining
                    newrows += 1
                    ghostmap[jlocal] = newrows
                    push!(data.j, N_local + ghostmap[jlocal])

                    owner = searchsortedlast(cs, jlocal)
                    push!(recvmap_new[owner], jlocal)
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

function schwarz!(
    sendmaps::Vector{Vector{Ti}},
    recvmaps::Vector{Vector{Ti}},
    ghostmaps::Dict{Ti,Ti},
    M::DistributedMatrix{Tv,Ti},
    overlap::Integer,
) where {Tv,Ti}
    comm      = M.comm
    commsize  = MPI.Comm_size(comm)
    myrank    = MPI.Comm_rank(comm)+1
    N_local   = size(M.loc, 1)
    nghosts   = size(M.int, 2)
    N_overlap = N_local + nghosts

    sendmaps .= copy.(M.sendmap)
    recvmaps .= copy.(M.recvmap)
    merge!(ghostmaps, M.ghostmap)

    I, J, V    = findnz(M.loc)
    Io, Jo, Vo = findnz(M.int)
    append!(I, Io)
    append!(J, Jo .+ N_local)
    append!(V, Vo)

    A = COOType{Tv,Ti}((I, J, V))

    loc_T = sparse(M.loc')
    int_T = sparse(M.int')
    outbufs = BufferType{Tv,Ti}((
        [Ti[] for _ ∈ 1:commsize],
        [Ti[] for _ ∈ 1:commsize],
        [Tv[] for _ ∈ 1:commsize],
    ))
    inbufs = BufferType{Tv,Ti}((
        [Ti[] for _ ∈ 1:commsize],
        [Ti[] for _ ∈ 1:commsize],
        [Tv[] for _ ∈ 1:commsize],
    ))
    offlocal = M.cs[myrank]-1
    send_counts = Vector{Int}(undef, commsize)

    sendmap_current = copy.(M.sendmap)
    recvmap_current = copy.(M.recvmap)
    recvmap_next    = [Ti[] for _ ∈ 1:commsize]

    for iter ∈ 1:overlap
        foreach(field -> foreach(empty!, field), outbufs)
        foreach(empty!, recvmap_next)
        fill!(send_counts, 0)
    addrows!(
        send_counts,
        outbufs,
        offlocal,
        loc_T,
        int_T,
        sendmap_current,
        M.ghost_global,
    )

    recv_counts = MPI.Alltoall(UBuffer(send_counts, 1), comm)
    for rank ∈ 1:commsize
        resize!(inbufs.i[rank], length(recvmap_current[rank]))
        resize!(inbufs.j[rank], recv_counts[rank])
        resize!(inbufs.v[rank], recv_counts[rank])
    end

    exchangerows!(inbufs, outbufs, comm)

    appendghostrows!(
        recvmap_next,
        A,
        inbufs,
        ghostmaps,
        recvmap_current,
        M.cs,
        M.cs[myrank],
        M.cs[myrank+1]-1,
        N_local,
        overlap - iter,
    )

    if iter < overlap
        sendmap_next = build_sendmap(recvmap_next, M.cs, comm)
        for r in 1:commsize
            append!(sendmaps[r], sendmap_next[r])
            append!(recvmaps[r], recvmap_next[r])
        end
        sendmap_current .= copy.(sendmap_next)
        recvmap_current .= copy.(recvmap_next)
    end

end

    N_overlap = N_local + length(ghostmaps)
    sparse(I, J, V, N_overlap, N_overlap)
end

struct RASPreconditioner{Tv,Ti,F}
    Pl::F
    N_local::Int
    buf::Vector{Tv}
    recv_sizes::Vector{Ti}
    sendmaps::Vector{Vector{Ti}}
    recvmaps::Vector{Vector{Ti}}
    sendbufs::Vector{Vector{Tv}}
    recvbufs::Vector{Vector{Tv}}
    ghostmaps::Dict{Ti,Ti}
    reqcount::Int
    comm::MPI.Comm
    function RASPreconditioner(
        M::DistributedMatrix{Tv,Ti},
        overlap = 2;
        τ = 1e-3,
    ) where {Tv,Ti}
        commsize = MPI.Comm_size(M.comm)
        sendmaps = [Ti[] for _ = 1:commsize]
        recvmaps = [Ti[] for _ = 1:commsize]
        ghostmaps = Dict{Ti,Ti}()
        S = schwarz!(sendmaps, recvmaps, ghostmaps, M, overlap)
        sendbufs = [Vector{Tv}(undef, length(buf)) for buf ∈ sendmaps]
        recvbufs = [Vector{Tv}(undef, length(buf)) for buf ∈ recvmaps]
        Pl = ilu(S; τ = τ)
        buf = Vector{Tv}(undef, size(S, 1))
        reqcount = count(!isempty, recvmaps) + count(!isempty, sendmaps)
        recv_sizes = length.(recvmaps)
        return new{Tv,Ti,ILUFactorization}(
            Pl,
            size(M.loc, 1),
            buf,
            recv_sizes,
            sendmaps,
            recvmaps,
            sendbufs,
            recvbufs,
            ghostmaps,
            reqcount,
            M.comm,
        )
    end
end

function ghostexchange!(P::RASPreconditioner, x::AbstractVector{Tv}) where {Tv}
    reqs = ghostexchange!(
        P.recvbufs,
        x,
        P.sendbufs,
        P.sendmaps,
        P.recv_sizes,
        P.reqcount,
        P.comm,
    )
    MPI.Waitall(reqs)
    commsize = length(P.recvmaps)
    for rank ∈ 1:commsize
        ids = P.recvmaps[rank]
        data = P.recvbufs[rank]
        for j ∈ eachindex(ids)
            jglobal = ids[j]
            jghost = P.ghostmaps[jglobal]
            P.buf[P.N_local+jghost] = data[j]
        end
    end
    return nothing
end

function LinearAlgebra.ldiv!(A::RASPreconditioner, b)
    copyto!(A.buf, 1, b, 1, length(b))

    ghostexchange!(A, b)

    ldiv!(A.Pl, A.buf)

    copyto!(b, 1, A.buf, 1, length(b))
end
