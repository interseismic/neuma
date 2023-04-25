mutable struct Adam
    theta::AbstractArray{Float32} # Parameter array
    m::AbstractArray{Float32}     # First moment
    v::AbstractArray{Float32}     # Second moment
    b1::Float32                   # Exp. decay first moment
    b2::Float32                   # Exp. decay second moment
    η::Float32                    # Step size
    eps::Float32                  # Epsilon for stability
    t::Int                        # Time step (iteration)
end

# Outer constructor
function Adam(theta::Array{Float32}, η=1f-3)
    m   = zeros(Float32, size(theta))
    v   = zeros(Float32, size(theta))
    b1  = 0.9f0
    b2  = 0.999f0
    eps = 1f-8
    t   = 0f0
    Adam(theta, m, v, b1, b2, η, eps, t)
end
  
function step!(opt::Adam, grads::Array{Float32})
    opt.t += 1f0
    gt    = grads
    opt.m = opt.b1 .* opt.m + (1f0 - opt.b1) .* gt
    opt.v = opt.b2 .* opt.v + (1f0 - opt.b2) .* gt .^ 2
    mhat = opt.m ./ (1f0 - opt.b1^opt.t)
    vhat = opt.v ./ (1f0 - opt.b2^opt.t)
    opt.theta -= opt.η .* (mhat ./ (sqrt.(vhat) .+ opt.eps))
end

# Outer constructor
function Adam(theta::AbstractArray{Float32}, η=1f-3)
    m   = CUDA.zeros(Float32, size(theta))
    v   = CUDA.zeros(Float32, size(theta))
    b1  = 0.9f0
    b2  = 0.999f0
    eps = 1f-8
    t   = 0f0
    Adam(theta, m, v, b1, b2, η, eps, t)
end
  
function step!(opt::Adam, grads::AbstractArray{Float32})
    opt.t += 1f0
    gt    = grads
    opt.m = opt.b1 .* opt.m + (1f0 - opt.b1) .* gt
    opt.v = opt.b2 .* opt.v + (1f0 - opt.b2) .* gt .^ 2
    mhat = opt.m ./ (1f0 - opt.b1^opt.t)
    vhat = opt.v ./ (1f0 - opt.b2^opt.t)
    opt.theta -= opt.η .* (mhat ./ (sqrt.(vhat) .+ opt.eps))
end