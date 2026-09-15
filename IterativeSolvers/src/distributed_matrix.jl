function scanghosts!(recvmap::Vector{Vector{Ti}}, A::SparseMatrixCSC{Tv,Ti}, cs::Vector{Int}, myrank::Int) where {Tv,Ti}
    ng      = 0
    ghosts  = Vector{Int}()
	startᵢ  = cs[myrank]
	stopᵢ   = cs[myrank+1] - 1
	@inbounds begin
        for j in 1:A.n
            if (j < startᵢ || j > stopᵢ) && 0 <= (ng += length(nzrange(A, j)))
                push!(ghosts, j)
            end
        end
        if !isempty(ghosts)
            firstneighbor = searchsortedlast(cs, ghosts[1])
            lastneighbor  = searchsortedlast(cs, ghosts[end])
            ghostsperrank = ceil(Int, length(ghosts) / (lastneighbor - firstneighbor + 1))
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

function build_sendmap(recvmap::Vector{Vector{Ti}}, cs::Vector{Int}, comm::MPI.Comm) where {Ti}
    commsize    = MPI.Comm_size(comm)
    recv_counts = length.(recvmap)
    send_counts = MPI.Alltoall(UBuffer(recv_counts, 1), comm)
    reqs = MPI.Request[]
    sendmap = [Ti[] for _ in 1:commsize]
	for (i, counts) ∈ enumerate(send_counts)
        buf = Vector{Ti}(undef, counts)
        if 0 <= counts
            req = MPI.Irecv!(buf, comm; source=i-1)
            push!(reqs, req)
        end
        sendmap[i] = buf
	end
    for (k, v) ∈ enumerate(recvmap)
        if !isempty(v)
            req = MPI.Isend(v, comm; dest=k-1)
            push!(reqs, req)
        end
	end
	MPI.Waitall(reqs)
    local_off = cs[myrank+1] - 1
    for v ∈ sendmap
        if !isempty(v)
            v .-= local_off
        end
    end
	return sendmap
end

function separate_local_ghosts(A::SparseMatrixCSC{Tv, Ti}, cs::Vector{Int}, nghosts::Int, myrank::Int) where {Tv, Ti}
	startᵢ = cs[myrank]
	stopᵢ = cs[myrank+1] - 1
	local_count = nnz(A) - nghosts
    Id          = sizehint!(Ti[], local_count)
    Jd          = sizehint!(Ti[], local_count)
    Vd          = sizehint!(Tv[], local_count)
    Io          = sizehint!(Ti[], nghosts)
    Jo          = sizehint!(Ti[], nghosts)
    Vo          = sizehint!(Tv[], nghosts)
    ghostsᵢ     = 0
	@inbounds for j ∈ 1:size(A, 2)
        if j >= start && j <= stop
            for k ∈ nzrange(A, j)
                push!(Id, A.rowval[k]) 
                push!(Jd, j - start  + 1)
                push!(Vd, A.nzval[k])
            end
        else
            ghostᵢ  += 1
            for k ∈ nzrange(A, j)
                push!(Io, A.rowval[k])
                push!(Jo, ghostᵢ)
                push!(Vo, A.nzval[k])
			end
		end
	end
	N_local = stop - start + 1

    Mₗ = sparse(Id, Jd, Vd, N_local, N_local)
    Mₒ = sparse(Io, Jo, Vo, N_ghost, N_local)
	return Mₗ, Mₒ
end

struct DistributedMatrix{Tv, Ti}
    loc::SparseMatrixCSC{Tv,Ti}
    int::SparseMatrixCSC{Tv,Ti}
    sendmap::Vector{Vector{Ti}}
    sendbuf::Vector{Vector{Tv}}
    ghosts::Vector{Tv}
    function DistributedMatrix(A::SparseMatrixCSC{Tv, Ti}, sizes::Vector{Ti}, perm_rows::Vector{Ti}, perm_cols::Vector{Ti}, comm::MPI.Comm) where {Tv, Ti}
        commsize = MPI.Comm_size(comm)
        myrank = MPI.Comm_rank(comm) + 1

        m = size(A,1)

        sizes = MPI.Allgather(m, comm)
        cs = cumsum([1; sizes])

        recvmap = [Ti[] for _ ∈ 1:commsize]
        ng      = scanghosts!(recvmap, A, cs, myrank)
        sendmap = build_sendmap(recvmap, cs, comm)
        
        M = separate_local_ghosts(A, cs, myrank)

        ghosts = Vector{Tv}(undef, size(Mₒ, 2))
        sendbufs = [Vector{Tv}(undef, length(sendmap[i])) for i in 1:commsize]

        return new{Tv, Ti}(recvmap, sendmap, Mₗ, Mₒ, ghosts, sendbufs, recvbufs)
    end
end

