module Eikonet
__precompile__
using CUDA
using CSV
using DataFrames
using Flux
using Flux.Data: DataLoader
using Flux: @epochs, mse, Params, update!
using IterTools: ncycle
using Plots
using Zygote
using LinearAlgebra
using BSON
using Interpolations: LinearInterpolation, Extrapolation
using Geodesy
using Distributions: Uniform
using Base.Iterators: flatten
using StatsBase: mean

include("./Input.jl")

abstract type VelocityModel end

struct VelMod1D <: VelocityModel
    df::DataFrame
    int_p::Extrapolation
    int_s::Extrapolation
end

function build_model()
    return Chain(
        Dense(7, 128, elu),
        Dense(128, 128, elu),
        Dense(128, 128, elu),
        Dense(128, 128, elu),
        Dense(128, 128, elu),
        Dense(128, 1, abs),
    )
end

# function build_model()
#     return Chain(
#         Dense(7, 32, elu),
#         SkipConnection(Chain(Dense(32, 32, elu), Dense(32, 32, elu)), +),
#         SkipConnection(Chain(Dense(32, 32, elu), Dense(32, 32, elu)), +),
#         SkipConnection(Chain(Dense(32, 32, elu), Dense(32, 32, elu)), +),
#         SkipConnection(Chain(Dense(32, 32, elu), Dense(32, 32, elu)), +),
#         # SkipConnection(Dense(32, 32, elu), +),
#         # SkipConnection(Dense(32, 32, elu), +),
#         # SkipConnection(Dense(32, 32, elu), +),
#         Dense(32, 1, abs),
#     )
# end

function build_linear_dataset(params, velmod::VelMod1D, n_train::Int, n_test::Int, batch_size::Int, device)
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
            v[i] = velmod.int_p(rec_depth[i]/Float32(1000))
        elseif phase_labels[i] == 1
            v[i] = velmod.int_s(rec_depth[i]/Float32(1000))
        else
            println(phase_labels[i])
            println("Error phase label not binary")
            return
        end
    end

    scaler = data_scaler(params)
    # StatsBase.transform!(scaler, x)
    forward!(x, scaler)

    x_train = x[:,1:n_train] |> device
    x_test = x[:,end-n_test+1:end] |> device
    y_train = reshape(v[1:n_train], 1, :) |> device
    y_test = reshape(v[end-n_test+1:end], 1, :) |> device

    train_data = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
    test_data = DataLoader((x_test, y_test), batchsize=batch_size, shuffle=true)
    
    return train_data, test_data
end

function FactoredEikonalPDE(x, model)
    x_src = x[1:3,:]
    x_rec = x[4:6,:]

    τ1 = model(x)
    τ0 = sqrt.(sum((x_rec - x_src).^2, dims=1))
    
    f(x) = sum(model(x))
    ∇τ1 = gradient(f, x)[1]
    ∇τ1 = ∇τ1[4:6,:]

    v1 = τ0.^2 .* sum(∇τ1.^2, dims=1)
    v2 = 2f0 .* τ1 .* sum(relu.((x_rec .- x_src) .* ∇τ1), dims=1)
    v3 = τ1.^2
    
    v̂ = (v1 .+ v2 .+ v3 .+ 1f-8) .^ (-0.5f0)
end

function EikonalPDE(x, model)
    x_src = x[1:3,:]
    x_rec = x[4:6,:]
    τ0 = sqrt.(sum((x_rec - x_src).^2, dims=1))

    f(x) = sum(τ0 .* model(x))
    ∇T = gradient(f, x)[1]

    v̂ = (sum(∇T[4:6,:] .^ 2, dims=1) .+ 1f-8) .^ (-0.5f0)
    return v̂
end

function EikonalLoss(x, v, model)
    # v̂ = EikonalPDE(x, model)
    v̂ = FactoredEikonalPDE(x, model)
    return sum(abs.(v̂ .- v) ./ v) / length(v)
end

function plot_solution(params, test_loader, model)
    x_test, v = test_loader.data
    scaler = data_scaler(params)
    # v̂ = EikonalPDE(x_test, model)
    v̂ = FactoredEikonalPDE(x_test, model)
    x_trans = inverse(x_test, scaler)
    scatter(x_trans[6,:]/1f3, v̂[1,:], label="v̂", left_margin = 20Plots.mm)
    scatter!(x_trans[6,:]/1f3, v[1,:], label="v", left_margin = 20Plots.mm)
    ylims!((0f0, 10.0))
    savefig("test_v.pdf")

    x_test = zeros(Float32, 7, 100)
    x_test[4,:] = collect(range(0f0, 1f0, length=100))
    T̂ = solve(x_test, model, scaler)
    x_trans = inverse(x_test, scaler)
    scatter(x_trans[4,:]/1f3, T̂[1,:], label="T̂", left_margin = 20Plots.mm)
    savefig("test_t.pdf")
end

function solve(x::AbstractArray{Float32}, model)
    x_src = x[1:3,:]
    x_rec = x[4:6,:]

    τ1 = model(x)
    τ0 = sqrt.(sum((x_rec - x_src).^2, dims=1))
    return τ0 .* τ1
end

function solve(x::AbstractArray{Float32}, model, scaler)
    if ndims(x) == 2
        τ0 = sqrt.(sum((x[4:6,:] - x[1:3,:]).^2, dims=1))
    elseif ndims(x) == 3
        τ0 = sqrt.(sum((x[4:6,:,:] - x[1:3,:,:]).^2, dims=1))
    end
    return τ0 .* model(x) .* scaler.scale ./ 1f3
end

function solve(x::AbstractArray, model, scaler)
    if ndims(x) == 2
        τ0 = sqrt.(sum((x[4:6,:] - x[1:3,:]).^2, dims=1))
    elseif ndims(x) == 3
        τ0 = sqrt.(sum((x[4:6,:,:] - x[1:3,:,:]).^2, dims=1))
    end
    return τ0 .* model(x) .* scaler.scale ./ 1f3
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
        train_loss, back = Zygote.pullback(() -> loss(d...), weights)
        grads = back(one(train_loss))
        update!(opt, weights, grads)
        push!(train_losses, train_loss)
    end
    for d in test_loader
        test_loss = loss(d...)
        push!(test_losses, test_loss)
    end
    return mean(train_losses), mean(test_losses)
end

function main(pfile; kws...)
    # params = JSON.parsefile(pfile)
    params = build_eikonet_syn_params()

    device = cpu
    # if has_cuda()		# Check if CUDA is available
    #     @info "CUDA is on"
    #     device = gpu
    #     CUDA.allowscalar(false)
    # end
    velmod = initialize_velmod(params, VelMod1D)

    loss(x, v) = EikonalLoss(x, v, model)
    
    # Construct model
    model = build_model()
    weights = Flux.params(model)
    opt = ADAM(params["lr"])

    println("Compiling model...")
    dummy_train, dummy_test = build_linear_dataset(params, velmod, 2, 2, 2, device)
    model = model |> device
    for (x,y) in dummy_train
        model(x |> device)
    end
    @time train_loss, test_loss = train_eikonet!(loss, weights,  dummy_train, dummy_test, opt)
    println("Finished compiling.")

    train_loader, test_loader = build_linear_dataset(params, velmod, 8192*8, 2048, 128, device)

    plot_solution(params, test_loader, model)

    println("Begin training Eikonet")
    # return
    for i in 1:params["n_epochs"]
        train_loss, test_loss = train_eikonet!(loss, weights, train_loader, test_loader, opt)
        println("Epoch $i ", train_loss, " ", test_loss)
    end

    BSON.@save params["model_file"] model
    # @load params["model_file"] model
    model = BSON.load(params["model_file"], @__MODULE__)[:model]

    plot_solution(params, test_loader, model)


end

end