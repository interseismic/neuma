

struct τ0{F} <: Lux.AbstractExplicitLayer
    scale::F
end

Lux.initialparameters(::AbstractRNG, ::τ0) = NamedTuple()

function Lux.initialstates(::AbstractRNG, l::τ0)
    return (scale=l.scale,)
end

struct EikoNet{L1, L2} <: Lux.AbstractExplicitContainerLayer{(:τ1, :τ0)}
    τ1::L1
    τ0::L2
end

function (eikonet::EikoNet)(x::AbstractArray, ps, st::NamedTuple)
    T0, st_τ0 = eikonet.τ0(x, ps.τ0, st.τ0)
    T1, st_τ1 = eikonet.τ1(x, ps.τ1, st.τ1)
    return T0 .* T1, (τ0 = st_τ0, τ1 = st_τ1)
end

function (l::τ0)(x::AbstractArray, ps, st::NamedTuple)
    T = st.scale * sqrt.(sum((x[4:6,:] - x[1:3,:]).^2, dims=1))
    return T, st
end

function EikonalPDE(eikonet::EikoNet, x::AbstractArray, ps, st::NamedTuple)
    τ0, _ = Lux.apply(eikonet.τ0, x, ps.τ0, st.τ0)
    τ1, _ = Lux.apply(eikonet.τ1, x, ps.τ1, st.τ1)
    ∇τ0 = (x[4:6,:] .- x[1:3,:]) ./ τ0
    f(x) = sum(eikonet.τ1(x, ps.τ1, st.τ1)[1])
    ∇τ1 = gradient(f, x)[1][4:6,:]
    ∇τ = τ1 .* ∇τ0 + τ0 .* ∇τ1
    s = sqrt.(sum(∇τ.^2, dims=1))
    return s
end