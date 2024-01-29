function convF16_depthwise_1D!(R, F, X)
    l = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    m = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z;
    
    D, _, M = size(F)
    C, _, N = size(X)
    L = size(R, 1)

    if l ≤ L && m ≤ M && n ≤ N
        val = 0
        for i in l:-1:(l-D+1)
            if i ≥ 1 && i ≤ C
                @inbounds val += X[i,m,n] * F[l+1-i,1,m]
            end
        end
        R[l,m,n] = val
    end
    return nothing
end

function convF16_1D!(R, F, X)
    l = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    m = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    D, _, M = size(F)
    C, _, N = size(X)
    L = size(R, 1)

    if l ≤ L && m ≤ M && n ≤ N
        val = 0
        for i in l:-1:(l-D+1)
            if i ≥ 1 && i ≤ C
                @inbounds val += X[i,1,n] * F[l+1-i,1,m]
            end
        end
        R[l,m,n] = val
    end
    return nothing
end

function convF16_depthwise_2D!(R, F, X)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    q = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    mn = (blockIdx().z - 1) * blockDim().z + threadIdx().z;
    H, W, _, M = size(F)
    C, E, _, N = size(X)
    P, Q, _, _ = size(R)

    if p ≤ P && q ≤ Q && mn ≤ M*N
        # infer m, n
        m, n = nothing, nothing
        if mn % M == 0
            m = M; n = mn ÷ M;
        else
            n = mn ÷ M + 1; m = mn - (n-1)*M;
        end

        val = 0
        for i in p:-1:(p-H+1)
            for j in q:-1:(q-W+1)
                if i < 1 || i > C || j < 1 || j > E
                else
                    val += X[i,j,m,n] * F[p+1-i,q+1-j,1,m]
                end
            end
        end
        R[p,q,m,n] = val        
    end
    return nothing
end

function convF16_2D!(R, F, X)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    q = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    mn = (blockIdx().z - 1) * blockDim().z + threadIdx().z;
    H, W, _, M = size(F)
    C, E, _, N = size(X)
    P, Q, _, _ = size(R)

    if p ≤ P && q ≤ Q && mn ≤ M*N
        # infer m, n
        m, n = nothing, nothing
        if mn % M == 0
            m = M; n = mn ÷ M;
        else
            n = mn ÷ M + 1; m = mn - (n-1)*M;
        end
        val = 0
        for i in p:-1:(p-H+1)
            for j in q:-1:(q-W+1)
                if i < 1 || i > C || j < 1 || j > E
                else
                    val += X[i,j,1,n] * F[p+1-i,q+1-j,1,m]
                end
            end
        end
        R[p,q,m,n] = val
    end
    return nothing
end

function crosscorF16_depthwise_1D!(R, F, X)
    l = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    m = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    D, _, M = size(F)
    C, _, N = size(X)
    L = size(R, 1)
    if l ≤ L && m ≤ M && n ≤ N
        val = 0
        for d in 1:D
            @inbounds val += X[l+d-1,m,n] * F[d,1,m]
        end
        R[l,m,n] = val
    end
    return nothing
end

function crosscorF16_1D!(R, F, X)
    l = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    m = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    n = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    D, _, M = size(F)
    C, _, N = size(X)
    L = size(R, 1)
    if l ≤ L && m ≤ M && n ≤ N
        val = 0
        for d in 1:D
            @inbounds val += X[l+d-1,1,n] * F[d,1,m]
        end
        R[l,m,n] = val
    end
    return nothing
end

function crosscorF16_depthwise_2D!(R, F, X)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    q = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    mn = (blockIdx().z - 1) * blockDim().z + threadIdx().z;
    H, W, _, M = size(F)
    C, E, _, N = size(X)
    P, Q, _, _ = size(R)

    if p ≤ P && q ≤ Q && mn ≤ M*N
        # infer m, n
        m, n = nothing, nothing
        if mn % M == 0
            m = M; n = mn ÷ M;
        else
            n = mn ÷ M + 1; m = mn - (n-1)*M;
        end
        val = 0
        for i in 1:H
            for j in 1:W
                @inbounds val += X[p+i-1,q+j-1,m,n] * F[i,j,1,m]
            end
        end
        R[p,q,m,n] = val
    end
    return nothing
end

function crosscorF16_2D!(R, F, X)
    p = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    q = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    mn = (blockIdx().z - 1) * blockDim().z + threadIdx().z;
    H, W, _, M = size(F)
    C, E, _, N = size(X)
    P, Q, _, _ = size(R)
    
    if p ≤ P && q ≤ Q && mn ≤ M*N
        # infer m, n
        m, n = nothing, nothing
        if mn % M == 0
            m = M; n = mn ÷ M;
        else
            n = mn ÷ M + 1; m = mn - (n-1)*M;
        end
        val = 0
        for i in 1:H
            for j in 1:W
                @inbounds val += X[p+i-1,q+j-1,1,n] * F[i,j,1,m]
            end
        end
        R[p,q,m,n] = val
    end
    return nothing
end

"""
note:
    F needs to have shape (D, 1, M)
    X needs to have shape (C, M, N)
    The result array will have shape (C+D-1, M, N)
"""
function conv_depthwise_1D(F, X; t=9)
    D, _, M = size(F)
    C, _, N = size(X)
    R = CUDA.zeros(eltype(F), (C+D-1, M, N));
    @cuda threads=(t,t,t) blocks=ceil.(Int, size(R) ./ t) convF16_depthwise_1D!(R, F, X)
    return R
end

function conv_1D(F, X; t=9)
    D, _, M = size(F)
    C, _, N = size(X)
    R = CUDA.zeros(eltype(F), (C+D-1, M, N));
    @cuda threads=(t,t,t) blocks=ceil.(Int, size(R) ./ t) convF16_1D!(R, F, X)
    return R
end

function conv_depthwise_2D(F, X; t=9)
    H, W, _, M = size(F)
    C, E, _, N = size(X)
    P, Q, MN = H+C-1, W+E-1, M*N
    R = CUDA.zeros(eltype(F), (P, Q, M, N));
    @cuda threads=(t,t,t) blocks=ceil.(Int, (P,Q,MN) ./ t) convF16_depthwise_2D!(R, F, X)
    return R
end

function conv_2D(F, X; t=9)
    H, W, _, M = size(F)
    C, E, _, N = size(X)
    P, Q, MN = H+C-1, W+E-1, M*N
    R = CUDA.zeros(eltype(F), (P, Q, M, N));
    @cuda threads=(t,t,t) blocks=ceil.(Int, (P,Q,MN) ./ t) convF16_2D!(R, F, X)
    return R
end

function crosscor_depthwise_1D(F, X; t=9)
    D, _, M = size(F)
    C, _, N = size(X)
    R = CUDA.zeros(eltype(F), (C-D+1, M, N));
    @cuda threads=(t,t,t) blocks=ceil.(Int, size(R) ./ t) crosscorF16_depthwise_1D!(R, F, X)
    return R
end

function crosscor_1D(F, X; t=9)
    D, _, M = size(F)
    C, _, N = size(X)
    R = CUDA.zeros(eltype(F), (C-D+1, M, N));
    @cuda threads=(t,t,t) blocks=ceil.(Int, size(R) ./ t) crosscorF16_1D!(R, F, X)
    return R
end

function crosscor_depthwise_2D(F, X; t=9)
    H, W, _, M = size(F)
    C, E, _, N = size(X)
    P, Q, MN = C-H+1,  E-W+1, M*N
    R = CUDA.zeros(eltype(F), (P, Q, M, N));
    @cuda threads=(t,t,t) blocks=ceil.(Int, (P,Q,MN) ./ t) crosscorF16_depthwise_2D!(R, F, X)
    return R
end

function crosscor_2D(F, X; t=9)
    H, W, _, M = size(F)
    C, E, _, N = size(X)
    P, Q, MN = C-H+1,  E-W+1, M*N
    R = CUDA.zeros(eltype(F), (P, Q, M, N));
    @cuda threads=(t,t,t) blocks=ceil.(Int, (P,Q,MN) ./ t) crosscorF16_2D!(R, F, X)
    return R
end