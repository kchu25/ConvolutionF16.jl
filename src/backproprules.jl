function conv_depthwise_1D_backprop_F_fillQ(Q, R̄, X)
    d = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    c = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    m = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    D, C, M = size(Q)
    N = size(X, 3)
    if d ≤ D && c ≤ C && m ≤ M
        val = 0
        for n in 1:N
            @inbounds val += R̄[d+c-1,m,n] * X[c,m,n]
        end
        Q[d,c,m] = val
    end
    return nothing
end

"""
An intermediate array Q is created to speed up this long filter cross-correlation
#TODO use FFT to speed this up
# modify this to fit memory requirements
"""
function conv_depthwise_1D_backprop_F(R̄, F, X; t=8)
    D, _, M = size(F)
    C, _, _ = size(X)
    Q = CUDA.zeros(eltype(F), (D,C,M));
    @cuda threads=(t,t,t) blocks=ceil.(Int, (D,C,M) ./ t) conv_depthwise_1D_backprop_F_fillQ(Q, R̄, X)
    return sum(Q, dims=2) # TODO is there a way to avoid this extra allocation?
end

function ChainRulesCore.rrule(::typeof(conv_depthwise_1D), F::CuArray, X::CuArray)
    R = conv_depthwise_1D(F, X)
    function conv_depthwise_1D_pullback(R̄)
        f̄ = NoTangent()
        F_bar = @thunk(conv_depthwise_1D_backprop_F(R̄, F, X))
        X_bar = @thunk(crosscor_depthwise_1D(F, R̄))
        return f̄, F_bar, X_bar
    end
    return R, conv_depthwise_1D_pullback
end

function conv_depthwise_2D_backprop_F_fillQ(Q, R̄, X)
    hw = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    ce = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    m = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    H, W, C, E, M = size(Q)
    N = size(X, 4)

    if hw ≤ H*W && ce ≤ C*E && m ≤ M
            # infer h and w
        if hw % H == 0 
            h = H; w = hw ÷ H;
        else
            w = hw ÷ H + 1; h = hw - (w-1)*H
        end
        # infer c and e
        if ce % C == 0
            c = C; e = ce ÷ C;
        else
            e = ce ÷ C + 1; c = ce - (e-1)*C
        end
        val = 0
        for n in 1:N
            @inbounds val += R̄[h+c-1,w+e-1,m,n] * X[c,e,m,n]
        end
        Q[h,w,c,e,m] = val
    end
    return nothing
end

function conv_depthwise_2D_backprop_F(R̄, F, X; t=8)
    H, W, _, M = size(F)
    C, E, _, N = size(X)
    Q = CUDA.zeros(eltype(F), (H, W, C, E, M));

    @cuda threads=(t,t,t) blocks=ceil.(Int, (H*W, C*E, M) ./ t) conv_depthwise_2D_backprop_F_fillQ(Q, R̄, X)
    return dropdims(sum(Q, dims=(3,4)),dims=4) # TODO is there a way to avoid this extra allocation?
end

function ChainRulesCore.rrule(::typeof(conv_depthwise_2D), F::CuArray, X::CuArray)
    R = conv_depthwise_2D(F, X)
    function conv_depthwise_2D_pullback(R̄)
        f̄ = NoTangent()
        F_bar = @thunk(conv_depthwise_2D_backprop_F(R̄, F, X))
        X_bar = @thunk(crosscor_depthwise_2D(F, R̄))
        return f̄, F_bar, X_bar
    end
    return R, conv_depthwise_2D_pullback
end

function conv_1D_backprop_F_fillQ(Q, R̄, X)
    d = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    c = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    m = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    D, C, M = size(Q)
    N = size(X, 3)
    if d ≤ D && c ≤ C && m ≤ M
        val = 0
        for n in 1:N
            @inbounds val += R̄[d+c-1,m,n] * X[c,1,n] # X[c,1,n] here, 1 is key
        end
        Q[d,c,m] = val
    end
    return nothing
end

function conv_1D_backprop_F(R̄, F, X; t=8)
    D, _, M = size(F)
    C, _, _ = size(X)
    Q = CUDA.zeros(eltype(F), (D,C,M));
    @cuda threads=(t,t,t) blocks=ceil.(Int, (D,C,M) ./ t) conv_1D_backprop_F_fillQ(Q, R̄, X)
    return sum(Q, dims=2)
end

function ChainRulesCore.rrule(::typeof(conv_1D), F::CuArray, X::CuArray)
    R = conv_1D(F, X)
    function conv_1D_pullback(R̄)
        f̄ = NoTangent()
        F_bar = @thunk(conv_1D_backprop_F(R̄, F, X))
        X_bar = @thunk(sum(crosscor_depthwise_1D(F, R̄),dims=2))
        # TODO is there a way to avoid this extra allocation sum(,dims=2)?
        return f̄, F_bar, X_bar
    end
    return R, conv_1D_pullback
end

function conv_2D_backprop_F_fillQ(Q, R̄, X)
    hw = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    ce = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    m = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    H, W, C, E, M = size(Q)
    N = size(X, 4)

    if hw ≤ H*W && ce ≤ C*E && m ≤ M
            # infer h and w
        if hw % H == 0 
            h = H; w = hw ÷ H;
        else
            w = hw ÷ H + 1; h = hw - (w-1)*H
        end
        # infer c and e
        if ce % C == 0
            c = C; e = ce ÷ C;
        else
            e = ce ÷ C + 1; c = ce - (e-1)*C
        end
        val = 0
        for n in 1:N
            @inbounds += R̄[h+c-1,w+e-1,m,n] * X[c,e,1,n]
        end
        Q[h,w,c,e,m] = val
    end
    return nothing
end

function conv_2D_backprop_F(R̄, F, X; t=8)
    H, W, _, M = size(F)
    C, E, _, N = size(X)
    Q = CUDA.zeros(eltype(F), (H, W, C, E, M));

    @cuda threads=(t,t,t) blocks=ceil.(Int, (H*W, C*E, M) ./ t) conv_2D_backprop_F_fillQ(Q, R̄, X)
    return dropdims(sum(Q, dims=(3,4)),dims=4) # TODO is there a way to avoid this extra allocation?
end

function ChainRulesCore.rrule(::typeof(conv_2D), F::CuArray, X::CuArray)
    R = conv_2D(F, X)
    function conv_2D_pullback(R̄)
        f̄ = NoTangent()
        F_bar = @thunk(conv_2D_backprop_F(R̄, F, X))
        X_bar = @thunk(sum(crosscor_2D(F, R̄),dims=3))
        return f̄, F_bar, X_bar
    end
    return R, conv_2D_pullback
end

function crosscor_depthwise_1D_backprop_F_fillQ(Q, R̄, X)
    d1 = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    d2 = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    m = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    D, _, M = size(Q)
    N = size(X, 3)
    if d1 ≤ D && d2 ≤ D && m ≤ M
        val = 0
        for n in 1:N
            @inbounds val += R̄[d2,m,n] * X[d2,m,n]
        end
        Q[d1,d2,m] = val
    end
    return nothing
end

function crosscor_depthwise_1D_backprop_F(R̄, F, X; t=8)
    D, _, M = size(F)
    C, _, _ = size(X)
    Q = CUDA.zeros(eltype(F), (D,D,M));
    @cuda threads=(t,t,t) blocks=ceil.(Int, (D,D,M) ./ t) crosscor_depthwise_1D_backprop_F_fillQ(Q, R̄, X)
    return sum(Q, dims=2)
end

function ChainRulesCore.rrule(::typeof(crosscor_depthwise_1D), F::CuArray, X::CuArray)
    R = crosscor_depthwise_1D(F, X)
    function crosscor_depthwise_1D_pullback(R̄)
        f̄ = NoTangent()
        F_bar = @thunk(crosscor_depthwise_1D_backprop_F(R̄, F, X))
        X_bar = @thunk(conv_depthwise_1D(F, R̄))
        return f̄, F_bar, X_bar
    end
    return R, crosscor_depthwise_1D_pullback
end

function crosscor_1D_backprop_F_fillQ(Q, R̄, X)
    d1 = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    d2 = (blockIdx().y - 1) * blockDim().y + threadIdx().y;
    m  = (blockIdx().z - 1) * blockDim().z + threadIdx().z;

    D, _, M = size(Q)
    N = size(X, 3)
    if d1 ≤ D && d2 ≤ D && m ≤ M
        val = 0
        for n in 1:N
            @inbounds val += R̄[d2,m,n] * X[d2,1,n]
        end
        Q[d1,d2,m] = val
    end
    return nothing
end

function crosscor_1D_backprop_F(R̄, F, X; t=8)
    D, _, M = size(F)
    C, _, _ = size(X)
    Q = CUDA.zeros(eltype(F), (D,D,M));
    @cuda threads=(t,t,t) blocks=ceil.(Int, (D,D,M) ./ t) crosscor_1D_backprop_F_fillQ(Q, R̄, X)
    return sum(Q, dims=2)
end

function ChainRulesCore.rrule(::typeof(crosscor_1D), F::CuArray, X::CuArray)
    R = crosscor_1D(F, X)
    function crosscor_1D_pullback(R̄)
        f̄ = NoTangent()
        F_bar = @thunk(crosscor_1D_backprop_F(R̄, F, X))
        X_bar = @thunk(sum(conv_depthwise_1D(F, R̄), dims=2))
        return f̄, F_bar, X_bar
    end
    return R, crosscor_1D_pullback
end
