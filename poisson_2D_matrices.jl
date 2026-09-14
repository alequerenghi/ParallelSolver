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
