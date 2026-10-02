
function getrange(N::Integer, myrank::Integer, commsize::Integer)
    q, r = divrem(N, commsize)
    start_idx = (myrank - 1) * q + min(myrank - 1, r) + 1
    len = q + (myrank <= r ? 1 : 0)
    return start_idx:(start_idx + len - 1)
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
    
    return sparse(I, J, V, N_local, N_global)
end
