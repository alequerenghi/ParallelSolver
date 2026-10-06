using SparseArrays

function Laplace_2D_5P(M::Int)
    a_P = 4*ones(M, M) # i   , j
    a_E =  -ones(M, M) # i+1 , j
    a_W =  -ones(M, M) # i-1 , j
    a_N =  -ones(M, M) # i   , j+1
    a_S =  -ones(M, M) # i   , j-1

    # BCs
    a_E[M,:] .= 0
    a_W[1,:] .= 0
    a_N[:,M] .= 0
    a_S[:,1] .= 0

    A = spdiagm(-M => a_S[:][(1+M):end  ],
                -1 => a_W[:][    2:end  ],
                 0 => a_P[:],
                 1 => a_E[:][    1:end-1],
                 M => a_N[:][    1:end-M])
    return dropzeros(A)
end


function Laplace_2D_9P(M::Int)
    a_P = 8*ones(M, M) # i   , j
    a_E =  -ones(M, M) # i+1 , j
    a_W =  -ones(M, M) # i-1 , j
    a_N =  -ones(M, M) # i   , j+1
    a_S =  -ones(M, M) # i   , j-1
    a_NE =  -ones(M, M) # i+1 , j
    a_NW =  -ones(M, M) # i-1 , j
    a_SE =  -ones(M, M) # i   , j+1
    a_SW =  -ones(M, M) # i   , j-1

    # BCs
    a_E[M,:] .= 0
    a_NE[M,:] .= 0
    a_SE[M,:] .= 0

    a_W[1,:] .= 0
    a_NW[1,:] .= 0
    a_SW[1,:] .= 0

    a_N[:,M] .= 0
    a_NE[:,M] .= 0
    a_NW[:,M] .= 0

    a_S[:,1] .= 0
    a_SE[:,1] .= 0
    a_SW[:,1] .= 0

    A = spdiagm(
             -(M+1) => a_SW[:][(1+(M+1)):end],
             - M    =>  a_S[:][(1+ M   ):end],
             -(M-1) => a_SE[:][(1+(M-1)):end],
                -1  => a_W[:][         2:end],
                 0  => a_P[:],
                 1  => a_E[:][ 1:(end-   1 )],
               M-1  => a_NW[:][1:(end-(M-1))],
               M    => a_N[:][ 1:(end- M   )],
               M+1  => a_NE[:][1:(end-(M+1))]
               )
    return dropzeros(A)
end

"""
    Laplace_2D_9P(M::Int, rows::UnitRange{Int}=1:M^2; T::Type=Float64)

Generates a submatrix containing only the specified `rows` of a 2D 9-point Laplacian 
sparse matrix for an M x M grid with Dirichlet boundary conditions.
"""
function Laplace_2D_9P(M::Int, rows::UnitRange{Int}; T::Type=Float64)
    n_rows = length(rows)
    r_start = first(rows)
    
    # Pre-allocate COO vectors (max 9 non-zeros per row)
    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, 9 * n_rows)
    sizehint!(J, 9 * n_rows)
    sizehint!(V, 9 * n_rows)

    for r in rows
        r_local = r - r_start + 1 # Row index in output submatrix
        
        # Convert 1D global index 'r' to 2D grid coordinates (i, j)
        i = (r - 1) % M + 1
        j = div(r - 1, M) + 1

        # Center (P)
        push!(I, r_local); push!(J, r); push!(V, T(8))

        # Direct Neighbors (E, W, N, S)
        if i < M; push!(I, r_local); push!(J, r + 1); push!(V, T(-1)); end # East
        if i > 1; push!(I, r_local); push!(J, r - 1); push!(V, T(-1)); end # West
        if j < M; push!(I, r_local); push!(J, r + M); push!(V, T(-1)); end # North
        if j > 1; push!(I, r_local); push!(J, r - M); push!(V, T(-1)); end # South

        # Diagonal Neighbors (NE, NW, SE, SW)
        if i < M && j < M; push!(I, r_local); push!(J, r + M + 1); push!(V, T(-1)); end # NE
        if i > 1 && j < M; push!(I, r_local); push!(J, r + M - 1); push!(V, T(-1)); end # NW
        if i < M && j > 1; push!(I, r_local); push!(J, r - M + 1); push!(V, T(-1)); end # SE
        if i > 1 && j > 1; push!(I, r_local); push!(J, r - M - 1); push!(V, T(-1)); end # SW
    end

    # Return sparse matrix of size (length(rows) x M^2)
    return sparse(I, J, V, n_rows, M^2)
end
