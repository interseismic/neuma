# module Eikonet
# __precompile__

using CSV
using DataFrames
using Flux
using Flux.Data: DataLoader
using Flux: @epochs, mse, Params, update!
using Plots
using Zygote
using LinearAlgebra
using BSON
using Random
using Interpolations: LinearInterpolation, Extrapolation
using Geodesy
using Distributions
using Base.Iterators: flatten
using StatsBase: mean

include("./Input.jl")
export MinmaxScaler, inverse, Origin, sec2date
export forward, data_scaler, get_stations, format_arrivals, generate_syn_dataset
export forward_point, inverse_point, EikoNet, EikoNet1D, solve

abstract type EikoNet end

struct EikoNet1D <: EikoNet
    model::Flux.Chain
    scale::Float32
end

function τ0(x::AbstractArray)
    if ndims(x) == 2
        return sqrt.(sum((x[4:6,:] - x[1:3,:]).^2, dims=1))
    elseif ndims(x) == 3
        return sqrt.(sum((x[4:6,:,:] - x[1:3,:,:]).^2, dims=1))
    elseif ndims(x) == 1
        return sqrt(sum((x[4:6] - x[1:3])^2))
    end
end

function ∇τ0(x::AbstractArray)
    return (x[4:6,:] .- x[1:3,:]) ./ τ0(x)
end

function (eikonet::EikoNet)(x::AbstractArray)
    τ1 = eikonet.model(x)
    return eikonet.scale * τ1 .* τ0(x)
end

function EikonalPDE(x::Array, eikonet::EikoNet1D)
    τ1 = eikonet.model(x)
    ∇τ1 = gradient(y -> sum(eikonet.model(y)), x)[1][4:6,:]
    ∇τ = τ1 .* ∇τ0(x) + τ0(x) .* ∇τ1
    s = sqrt.(sum(∇τ.^2, dims=1))
end

function build_model()
    return Chain(
        Dense(7, 64, elu),
        SkipConnection(Dense(64, 64, elu), +),
        SkipConnection(Dense(64, 64, elu), +),
        SkipConnection(Dense(64, 64, elu), +),
        SkipConnection(Dense(64, 64, elu), +),
        Dense(64, 1, abs),
    )
end

# function build_model()
#     return Chain(
#         Dense(7, 1),
#     )
# end

Flux.@functor EikoNet1D (model,)

abstract type VelocityModel end

function forward(X::Array{Float32}, eikonet::EikoNet)
    return forward(X, eikonet.scale)
end

function forward(X::Array, eikonet::EikoNet)
    return forward(X, eikonet.scale)
end

function inverse(Y::AbstractArray, eikonet::EikoNet)
    return inverse(Y, eikonet.scale)
end

struct VelMod1D <: VelocityModel
    df::DataFrame
    int_p::Extrapolation
    int_s::Extrapolation
end

function build_linear_dataset(params, velmod::VelMod1D, n_train::Int, n_test::Int, batch_size::Tuple, device)
    n_tot = n_train + n_test
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    src_lat = rand(Uniform(params["lat_min"], params["lat_max"]), n_tot)
    src_lon = rand(Uniform(params["lon_min"], params["lon_max"]), n_tot)
    src_depth = rand(Uniform(params["z_min"], params["z_max"]), n_tot) .* 1f3
    sources = trans.([LLA(lat=src_lat[i], lon=src_lon[i]) for i in 1:n_tot])

    rec_lat = rand(Uniform(params["lat_min"], params["lat_max"]), n_tot)
    rec_lon = rand(Uniform(params["lon_min"], params["lon_max"]), n_tot)
    rec_depth = rand(Uniform(params["z_min"], params["z_max"]), n_tot) .* 1f3
    recs = trans.([LLA(lat=rec_lat[i], lon=rec_lon[i]) for i in 1:n_tot])    

    v = zeros(Float32, n_tot)
    x = zeros(Float32, 7, n_tot)
    phase_labels = collect(1:n_tot) .% 2
 
    for i in 1:n_tot
        x[1,i] = sources[i].e
        x[2,i] = sources[i].n
        x[3,i] = src_depth[i]

        x[4,i] = recs[i].e
        x[5,i] = recs[i].n
        x[6,i] = rec_depth[i]
        x[7,i] = phase_labels[i]

        if phase_labels[i] == 0
            v[i] = 1f0/velmod.int_p(rec_depth[i]/Float32(1000))
        elseif phase_labels[i] == 1
            v[i] = 1f0/velmod.int_s(rec_depth[i]/Float32(1000))
        else
            println(phase_labels[i])
            println("Error phase label not binary")
            return
        end
    end

    x[1:6,:] ./= 1f3
    # x[1:6,:] .-= minimum(x[1:6,:], dims=2)
    # x[1:6,:] ./= maximum(x[1:6,:])
    scaler = data_scaler(params)
    x = Array{Float32}(forward(x, scaler))

    # println(minimum(x, dims=2))
    # println(maximum(x, dims=2))

    x_train = x[:,1:n_train] |> device
    x_test = x[:,end-n_test+1:end] |> device
    y_train = reshape(v[1:n_train], 1, :) |> device
    y_test = reshape(v[end-n_test+1:end], 1, :) |> device

    train_data = DataLoader((x_train, y_train), batchsize=batch_size[1], shuffle=true)
    test_data = DataLoader((x_test, y_test), batchsize=batch_size[2], shuffle=true)
    
    return train_data, test_data
end

function FactoredEikonalPDE(x::AbstractArray, model, scaler)
    x_const = forward(x, scaler)
    x_src = x[1:3,:]
    x_rec = x[4:6,:]

    τ1 = model(x_const)
    τ0 = sqrt.(sum((x_rec - x_src).^2, dims=1))
    
    f(x) = sum(model(x))
    ∇τ1 = gradient(f, x_const)[1]
    ∇τ1 = ∇τ1[4:6,:]

    v1 = τ0.^2 .* sum(∇τ1.^2, dims=1)
    v2 = 2f0 .* τ1 .* sum((x_rec .- x_src) .* ∇τ1, dims=1)
    v3 = τ1.^2
    
    ŝ = sqrt.(v1 .+ v2 .+ v3)
    return ŝ
end

function EikonalLoss(x::AbstractArray, s::AbstractArray, model, scaler)
    ŝ = FactoredEikonalPDE(x, model, scaler)
    # ŝ = EikonalPDE(x, model, scaler)
    return Flux.mse(ŝ, s) + Flux.mae(ŝ, s)
end

function EikonalLoss(x::AbstractArray, s::AbstractArray, eikonet::EikoNet1D)
    ŝ = EikonalPDE(x, eikonet)
    return Flux.mse(ŝ, s) + Flux.mae(ŝ, s)
end

function NormalizedEikonalLoss(x::Array, v::Array, model, scaler)
    v̂ = FactoredEikonalPDE(x, model, scaler)
    return mean(((v̂ .- v) ./ v).^2)
end

function plot_solution(params, test_loader, model::Chain)
    x_test, s = test_loader.data
    scaler = data_scaler(params)
    ŝ = FactoredEikonalPDE(x_test, model, scaler)
    v̂ = 1f0 ./ ŝ
    v = 1f0 ./ s
    # x_cart = inverse(x_test, scaler)
    x_cart = x_test
    scatter(x_cart[6,:], v̂[1,:], label="v̂", left_margin = 20Plots.mm)
    scatter!(x_cart[6,:], v[1,:], label="v", left_margin = 20Plots.mm)
    ylims!((0f0, 10.0))
    savefig("test_v.pdf")

    x_test = zeros(Float32, 7, 100)
    x_test[3,:] .= Float32(0.10)
    x_test[4,:] = collect(range(0f0, 1f0, length=100))
    x_cart = inverse(x_test, scaler)
    T̂ = solve(x_cart, model, scaler)
    scatter(x_cart[4,:], T̂[1,:], label="T̂", left_margin = 20Plots.mm)
    savefig("test_t.pdf")
end

function plot_solution(params, test_loader, eikonet::EikoNet)
    x_test, s = test_loader.data
    scaler = data_scaler(params)
    ŝ = EikonalPDE(x_test, eikonet)
    x_test = inverse(x_test, scaler)
    v̂ = 1f0 ./ ŝ
    v = 1f0 ./ s
    # x_cart = inverse(x_test, scaler)
    scatter(x_test[6,:], v̂[1,:], label="v̂", left_margin = 20Plots.mm)
    scatter!(x_test[6,:], v[1,:], label="v", left_margin = 20Plots.mm)
    ylims!((0f0, 10.0))
    savefig("test_v.pdf")

    x_test = zeros(Float32, 7, 100)
    x_test[3,:] .= Float32(15.0)
    x_test[4,:] = collect(range(0f0, 200f0, length=100))
    x_test = forward(x_test, scaler)
    T̂ = eikonet(x_test)
    x_test = inverse(x_test, scaler)
    scatter(x_test[4,:], T̂[1,:], label="T̂", left_margin = 20Plots.mm)
    savefig("test_t.pdf")
end

function solve(x::AbstractArray, model::Chain, scaler::MinmaxScaler)
    if ndims(x) == 2
        x_const = forward(x, scaler)
        τ0 = sqrt.(sum((x[4:6,:] - x[1:3,:]).^2, dims=1))
    elseif ndims(x) == 3
        x_const = forward(x, scaler)
        τ0 = sqrt.(sum((x[4:6,:,:] - x[1:3,:,:]).^2, dims=1))
    end
    return τ0 .* model(x_const)
end

function solve(x::AbstractArray, eikonet::EikoNet)
    return solve(x, eikonet.model, eikonet.scaler)
end

function initialize_velmod(params, ::Type{VelMod1D})
    df = CSV.read(params["velmod_file"], DataFrame)
    int_p = LinearInterpolation(df.depth, df.vp)
    int_s = LinearInterpolation(df.depth, df.vs)
    return VelMod1D(df, int_p, int_s)
end

function train_eikonet!(loss, weights, train_loader, test_loader, opt)
    #weights = Flux.Params(weights)
    train_losses = Vector{Float32}()
    test_losses = Vector{Float32}()
    for d in train_loader
        train_loss, grads = Zygote.withgradient(() -> loss(d...), weights)
        update!(opt, weights, grads)
        push!(train_losses, train_loss)
    end
    for d in test_loader
        test_loss = loss(d...)
        push!(test_losses, test_loss)
    end
    return mean(train_losses), mean(test_losses)
end

function train(pfile; kws...)
    params = JSON.parsefile(pfile)

    device = cpu
    velmod = initialize_velmod(params, VelMod1D)

    scaler = data_scaler(params)

    # Construct model
    # model = build_model()
    # model = fmap(f64, model)
    # weights = Flux.params(model)

    eikonet = EikoNet1D(build_model(), scaler.scale)
    # opt = ADAM(params["lr"])
    opt_state = Flux.setup(ADAM(params["lr"]), eikonet)

    println("Compiling model...")
    n_train = 1000000
    n_test = 100000
    dummy_train, dummy_test = build_linear_dataset(params, velmod, 2, 2, (2, 2), device)
    train_loader, test_loader = build_linear_dataset(params, velmod, n_train, n_test, (32, 1024), device)
    # plot_solution(params, test_loader, eikonet)

    @time for data in dummy_train
        x, s = data
        val, grads = Flux.withgradient(eikonet) do m
            ŝ = EikonalPDE(x, m)
            mean(abs.((ŝ - s) ./ s))
        end
        Flux.update!(opt_state, eikonet, grads[1])
    end
    println("Finished compiling.")

    println("Begin training Eikonet")
    loss_best = Inf
    for i in 1:params["n_epochs"]
        train_loss = 0f0
        test_loss = 0f0
        for data in train_loader
            x, s = data
            val, grads = Flux.withgradient(eikonet) do m
                ŝ = EikonalPDE(x, m)
                0.9 * mean(abs.((ŝ - s) ./ s)) + 0.1 * mean(abs2.((ŝ - s) ./ s))
            end
            Flux.update!(opt_state, eikonet, grads[1])
            train_loss += val * length(s)
        end
        train_loss /= n_train
        for data in test_loader
            x, s = data
            ŝ = EikonalPDE(x, eikonet)
            val = 0.9 * mean(abs.((ŝ - s) ./ s)) + 0.1 * mean(abs2.((ŝ - s) ./ s))
            test_loss += val * length(s)
        end
        test_loss /= n_test
        if test_loss < loss_best
            BSON.@save params["model_file"] eikonet
            loss_best = test_loss
        end
        println("Epoch $i train $train_loss test $test_loss best $loss_best")
        plot_solution(params, test_loader, eikonet)
    end

    plot_solution(params, test_loader, eikonet)

end

function plot_results(pfile)
    params = JSON.parsefile(pfile)
    eikonet = BSON.load(params["model_file"], @__MODULE__)[:eikonet]
    velmod = initialize_velmod(params, VelMod1D)
    train_loader, test_loader = build_linear_dataset(params, velmod, 2000, 2000, (64, 64), cpu)
    scaler = data_scaler(params)
    plot_solution(params, test_loader, eikonet)
end

# end
