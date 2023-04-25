module MAPDD
__precompile__

using DataFrames
using CSV
using JSON
using Zygote
using Flux
using Geodesy
using BSON
using Dates
using StatsBase
using Plots
using Combinatorics
using LinearAlgebra
using Distributions
using NearestNeighbors
using Distributed
using ProgressMeter
using SparseArrays
using BenchmarkTools
using CUDA

include("./Input.jl")
include("./Eikonet.jl")
include("./Adam.jl")

abstract type InversionMethod end
abstract type MAP <: InversionMethod end
abstract type SVI <: InversionMethod end

function format_input(stations, origins0, trans)
    n_sta = nrow(stations)
    n_events = nrow(origins0)
    n_pred = 2 * n_events * n_sta

    # X_inp needs to have dim of [n_sta, 3]
    X_sta = cat(stations.X, stations.Y, stations.Z, dims=2)

    # X_src has dimensions [n_events, 4]
    vec_lla = Vector{LLA{Float64}}()
    for row in eachrow(origins0)
        lla = LLA(lat=row.lat, lon=row.lon)
        push!(vec_lla, lla)
    end
    vec_enu = map(trans, vec_lla)
    X_src = zeros(Float32, nrow(origins0), 4)
    # Note that initial origin times are zero (centered on otime)
    for i in 1:nrow(origins0)
        X_src[i,2] = vec_enu[i].e
        X_src[i,3] = vec_enu[i].n
        X_src[i,4] = origins0.depth[i]*1f3
    end

    X = zeros(Float32, n_pred, 8)
    map_3d2d = zeros(Int32, n_pred, 3)
    map_all2row = Dict()
    count = 1
    phase_label = ["P", "S"]
    for i in 1:n_events
        for j in 1:n_sta
            for k in 1:2
                X[count,1:4] .= X_src[i,:]
                X[count,5:7] .= X_sta[j,:]
                X[count,8] = k - 1.0
                map_3d2d[count,:] .= [i,j,k]
                map_all2row[(origins0.evid[i], stations.network[j], stations.station[j], phase_label[k])] = Int32(count)
                count += 1
            end
        end
    end
    return X, map_3d2d, map_all2row
end

function format_input2(stations, origins0, trans)
    n_sta = nrow(stations)

    map_all2row = Dict()
    count = 1
    X_sta = zeros(Float32, 4, n_sta*2)
    phase2int = Dict("P"=>0f0, "S"=>1f0)
    for phase in ["P", "S"]
        for row in eachrow(stations)
            map_all2row[(row.network, row.station, phase)] = Int32(count)
            X_sta[1,count] = row.X
            X_sta[2,count] = row.Y
            X_sta[3,count] = row.Z
            X_sta[4,count] = phase2int[phase]
            count += 1
        end
    end

    # X_src has dimensions [n_events, 4]
    vec_lla = Vector{LLA{Float64}}()
    for row in eachrow(origins0)
        lla = LLA(lat=row.latitude, lon=row.longitude)
        push!(vec_lla, lla)
    end
    vec_enu = map(trans, vec_lla)
    X_src = zeros(Float32, 4, nrow(origins0))
    # Note that initial origin times are zero (centered on otime)
    map_evid2row = Dict()
    for (i, row) in enumerate(eachrow(origins0))
        X_src[2,i] = vec_enu[i].e
        X_src[3,i] = vec_enu[i].n
        X_src[4,i] = row.depth*1f3
        map_evid2row[row.evid] = i
    end

    return X_sta, X_src, map_all2row, map_evid2row
end

function selectormatrix(dtimes::DataFrame, map_all2row::Dict, n_pred::Int)
    # Loop over rows of dtimes, use evid1, evid2, phase, and sta to get [i,j]
    I = zeros(Int32, nrow(dtimes))
    J = zeros(Int32, nrow(dtimes))
    V = zeros(Float32, nrow(dtimes))
    for (i, row) in enumerate(eachrow(dtimes))
        col1 = map_all2row[(row.evid1, row.network, row.station, row.phase)]
        I[i] = i
        J[i] = col1
        V[i] = 1f0
    end
    Q1 = sparse(I, J, V, nrow(dtimes), n_pred)
    for (i, row) in enumerate(eachrow(dtimes))
        col2 = map_all2row[(row.evid2, row.network, row.station, row.phase)]
        I[i] = i
        J[i] = col2
        V[i] = 1f0
    end
    Q2 = sparse(I, J, V, nrow(dtimes), n_pred)
    return Q1, Q2
end

function projectionmatrix(origins::DataFrame, map_3d2d::Array)
    I = zeros(Int32, size(map_3d2d, 1))
    J = zeros(Int32, size(map_3d2d, 1))
    V = zeros(Float32, size(map_3d2d, 1))
    for i in 1:size(map_3d2d, 1)
        evid_row = map_3d2d[i,1]
        I[i] = Int32(evid_row)
        J[i] = Int32(i)
        V[i] = 1f0
    end
    A = sparse(I, J, V, nrow(origins), size(map_3d2d,1))
    A = A ./ sum(A, dims=2)
    return A
end


function averagingmatrix(origins::DataFrame, dtimes::DataFrame, map3d2d::Array, map_all2row::Dict)
    I = zeros(Int32, nrow(dtimes))
    J = zeros(Int32, nrow(dtimes))
    V = zeros(Float32, nrow(dtimes))
    for (i, row) in enumerate(eachrow(dtimes))
        evid_row = map_all2row[(row.evid1, row.network, row.station, row.phase)]
        evid_row = map3d2d[evid_row,1]
        I[i] = evid_row
        J[i] = i
        V[i] = 1f0
    end
    A1 = sparse(I, J, V, nrow(origins), nrow(dtimes))
    A01 = A1 * ones(Float32, nrow(dtimes))

    for (i, row) in enumerate(eachrow(dtimes))
        evid_row = map_all2row[(row.evid2, row.network, row.station, row.phase)]
        evid_row = map3d2d[evid_row,1]
        I[i] = evid_row
        J[i] = i
        V[i] = 1f0
    end
    A2 = sparse(I, J, V, nrow(origins), nrow(dtimes))
    A02 = A2 * ones(Float32, nrow(dtimes))

    A0 = A01 + A02

    A0 = reshape(A0, nrow(origins), 1)
    # A1 ./= A0
    # A2 ./= A0

    idx = findall(isnan.(A1))
    A1[idx] .= 0f0
    idx = findall(isnan.(A2))
    A2[idx] .= 0f0
    return A1, A2, A0
end

function liftingmatrix(origins::DataFrame, map_3d2d::Array)
    I = zeros(Int32, size(map_3d2d, 1))
    J = zeros(Int32, size(map_3d2d, 1))
    V = zeros(Float32, size(map_3d2d, 1))
    Threads.@threads for i in 1:size(map_3d2d, 1)
        evid_row = map_3d2d[i,1]
        I[i] = i
        J[i] = evid_row
        V[i] = 1f0
    end
    P = sparse(I, J, V, size(map_3d2d,1), nrow(origins))
    return P
end

function build_station2row_map(stations)
    map_sta2row = Dict()
    for (i, row) in enumerate(eachrow(stations))
        map_sta2row[(row.network, row.station)] = i
    end
    return map_sta2row
end

function build_evid2row_map(origins)
    map_evid2row = Dict()
    for (i, row) in enumerate(eachrow(origins))
        map_evid2row[row.evid] = i
    end
    return map_evid2row
end

function swap(x::AbstractArray)
    permutedims(x, (1,3,2))
end

function locate_old(params, origins0, dtimes, stations, eikonet, scaler)
    η = Float32(params["lr"])
    n_sta = nrow(stations)
    n_events = nrow(origins0)
    n_pred = 2 * n_events * n_sta
    ΔT_obs = Vector{Float32}(dtimes.dt)

    trans = ENUfromLLA(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    println("Building initial X matrix")
    X, map_3d2d, map_all2row = format_input(stations, origins0, trans)
    X = Array(transpose(X))
    X[2:end,:] = forward(X[2:end,:], scaler)

    println("Building sparse matrices [0]")
    Q1, Q2 = selectormatrix(dtimes, map_all2row, n_pred)
    println("Building sparse matrices [1]")
    A1, A2, A0 = averagingmatrix(origins0, dtimes, map_3d2d, map_all2row)

    println("Splitting X in half")
    X = cat((Q1 * X')', (Q2 * X')', dims=3)
    Q1 = Q2 = nothing
    Xinit = gpu(A1 * X[1:4,:,1]' + A2 * X[1:4,:,2]')

    println("Transferring to gpu")
    eikonet = gpu(eikonet)
    X = gpu(X)
    # A1 = gpu(A1)
    # A2 = gpu(A2)
    A0 = gpu(A0)
    ΔT_obs = gpu(ΔT_obs)
    indexes = Array{Int32}(1:length(ΔT_obs))

    likelihood_dist = Laplace(0f0, params["phase_unc"])
    # For prior, scale param is in meters
    priordist = Vector{Distribution}()
    push!(priordist, Normal(0f0, Float32(2.0)))
    push!(priordist, Normal(0f0, Float32(5000/scaler.scale)))
    push!(priordist, Normal(0f0, Float32(5000/scaler.scale)))
    push!(priordist, Normal(0f0, Float32(5000/scaler.scale)))
    likelihood_dist = gpu(likelihood_dist)
    priordist = gpu(priordist)

    println()
    println("Begin gradient descent")
    loader = Flux.DataLoader((swap(X), ΔT_obs, indexes), batchsize=min(2^16, nrow(dtimes)))

    function likelihoodloss(X::AbstractArray{Float32}, ΔT_obs::AbstractArray{Float32})      
        T_pred = dropdims(Eikonet.solve(X[2:end,:,:], eikonet, scaler), dims=1) .+ X[1,:,:]
        ΔT_pred = T_pred[:,2] - T_pred[:,1]
        -sum(logpdf.(likelihood_dist, ΔT_obs - ΔT_pred))
    end
    function priorloss(Xnew::AbstractArray{Float32})
        loss = 0f0
        for i in 1:4
            loss += sum(logpdf.(priordist[i], Xnew[i,:] - Xinit[i,:]))
        end
        return -loss
    end

    opt1 = Adam(X[1:4,:,1], η)
    opt2 = Adam(X[1:4,:,2], η)

    for i in 1:params["n_epochs"]

        # # First compute prior and its gradient
        # Xnew = A1 * loader.data[1][1:4,1,:]' + A2 * loader.data[1][1:4,2,:]'
        # Xnew = gpu(Xnew)
        # px, ∇px = withgradient(priorloss, Xnew)
        # ∇px = ∇px[1]

        # Next compute likelihood and its gradient
        LL = 0f0
        ∇LL = gpu(zeros(Float32, size(A1, 1), 4))
        for (x, y, idx) in loader
            LL0, ∇LL0 = withgradient(xx -> likelihoodloss(xx, y), swap(x))
            LL += LL0
            ∇LL0 = ∇LL0[1][1:4,:,:]
            A1sub = gpu(A1[:,idx])
            A2sub = gpu(A2[:,idx])
            ∇LL += (A1sub*∇LL0[:,:,1]' + A2sub*∇LL0[:,:,2]')
        end

        px = 0f0
        ∇px = 0f0
        loss = (LL + px) ./ length(ΔT_obs)
        ∇L = (∇LL .+ ∇px) ./ length(ΔT_obs)
        ∇L = cpu(∇L)
        println("Iter: $i, loss: $loss, LL: $LL") #, px: $px")
        
        step!(opt1, gpu(Array((A1' * ∇L)')))
        step!(opt2, gpu(Array((A2' * ∇L)')))
        X[1:4,:,1] .= opt1.theta
        X[1:4,:,2] .= opt2.theta
        # X[1:4,:,1] .-= η .* gpu(A1' * ∇L)'
        # X[1:4,:,2] .-= η .* gpu(A2' * ∇L)'
        X[findall(X[2:4,:,:] .< 0f0)] .= 0f0 #Clipping at boundary
        X[findall(X[2:4,:,:] .> 1f0)] .= 1f0
        loader.data[1] .= swap(X)
    end
    println("Gradient descent complete")
    println()

    X = swap(loader.data[1])
    X = cpu(permutedims(X, (2,1,3)))
    X = ((A1*X[:,:,1] + A2*X[:,:,2]) ./ cpu(A0))'
    X[2:end,:] = inverse(X[2:end,:], scaler)

    origins = DataFrame(time=DateTime[], evid=String[], lat=Float32[], lon=Float32[],
                        depth=Float32[], mag=Float32[], X=Float32[], Y=Float32[])
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    inv_trans = LLAfromENU(origin, wgs84)
    for (i, row) in enumerate(eachrow(origins0))
        origin_time = row.time + sec2date(X[1,1])
        hypo_lla = inv_trans(ENU(X[2,i], X[3,i], 0f0))
        push!(origins, (origin_time, row.evid, hypo_lla.lat, hypo_lla.lon,
                        X[4,i]/1f3, row.mag, X[2,i]/1f3, X[3,i]/1f3))
    end

    return origins
end

function locate(params, origins0, dtimes, stations, eikonet, scaler)
    η = Float32(params["lr"])
    n_obs = nrow(dtimes)
    ΔT_obs = Vector{Float32}(dtimes.dt)

    trans = ENUfromLLA(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    println("Building initial X matrix")
    X_sta, X_src, map_all2row, map_evid2row = format_input2(stations, origins0, trans)
    X_sta[1:3,:] .=  (X_sta[1:3,:] .- scaler.min) ./ scaler.scale
    X_src[2:4,:] .=  (X_src[2:4,:] .- scaler.min) ./ scaler.scale

    I = zeros(Int, 4, nrow(dtimes))
    println("Building index maps")
    Threads.@threads for i in 1:nrow(dtimes)
        I[1,i] = map_evid2row[dtimes.evid1[i]]
        I[2,i] = map_evid2row[dtimes.evid2[i]]
        I[3,i] = map_all2row[(dtimes.network[i], dtimes.station[i], dtimes.phase[i])]
        I[4,i] = i
    end

    println("Transferring to gpu")
    eikonet = gpu(eikonet)
    X_src = gpu(X_src)
    X_init = copy(X_src)
    X_sta = gpu(X_sta)

    likelihood_dist = Laplace(0f0, params["phase_unc"])
    # For prior, scale param is in meters
    priordist = Vector{Distribution}()
    push!(priordist, Normal(0f0, Float32(0.5)))
    push!(priordist, Normal(0f0, Float32(500/scaler.scale)))
    push!(priordist, Normal(0f0, Float32(500/scaler.scale)))
    push!(priordist, Normal(0f0, Float32(500/scaler.scale)))
    likelihood_dist = gpu(likelihood_dist)
    priordist = gpu(priordist)

    println()
    println("Creating dataloader")
    loader = Flux.DataLoader((I, ΔT_obs), batchsize=min(params["batch_size"], nrow(dtimes)))

    opt = Adam(X_src, η)

    println("Begin gradient descent")
    for i in 1:params["n_epochs"]
        function loss(X_src::AbstractArray{Float32}, indexes::AbstractArray{Int}, ΔT_obs::AbstractArray{Float32})
            x1 = cat(X_src[:,indexes[1,:]], X_sta[:,indexes[3,:]], dims=1)
            x2 = cat(X_src[:,indexes[2,:]], X_sta[:,indexes[3,:]], dims=1)
            X = cat(x1, x2, dims=3)
            T_pred = dropdims(Eikonet.solve(X[2:end,:,:], eikonet, scaler), dims=1) .+ X[1,:,:]
            ΔT_pred = T_pred[:,2] - T_pred[:,1]
            logL = sum(logpdf.(likelihood_dist, ΔT_obs - ΔT_pred))
            # logpx = sum([sum(logpdf.(priordist[i], X_src[i,:] - X_init[i,:])) for i in 1:4])
            return -logL #- logpx
        end

        L = 0f0
        ∇L = CUDA.zeros(Float32, size(X_src))
        runtime = @elapsed for (indexes, y) in CuIterator(loader)
            L0, ∇L0 = withgradient(x -> loss(x, indexes, y), X_src)
            L += L0 / n_obs 
            ∇L += ∇L0[1] / n_obs
        end

        step!(opt, ∇L)
        # CUDA.unsafe_free!(∇L)
        X_src = opt.theta
        # X_src[findall(X_src[2:4,:] .< 0f0)] .= 0f0 #Clipping at boundary
        # X_src[findall(X_src[2:4,:] .> 1f0)] .= 1f0
        println("Iter $i loss: $L ($runtime sec)")
    end
    println("Gradient descent complete")
    println()

    X = cpu(X_src)
    X[2:4,:] = (X[2:4,:] .* scaler.scale) .+ scaler.min

    origins = DataFrame(time=DateTime[], evid=String[], lat=Float32[], lon=Float32[],
                        depth=Float32[], mag=Float32[], X=Float32[], Y=Float32[])
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    inv_trans = LLAfromENU(origin, wgs84)
    for (i, row) in enumerate(eachrow(origins0))
        origin_time = row.time + sec2date(X[1,1])
        hypo_lla = inv_trans(ENU(X[2,i], X[3,i], 0f0))
        push!(origins, (origin_time, row.evid, hypo_lla.lat, hypo_lla.lon,
                        X[4,i]/1f3, NaN, X[2,i]/1f3, X[3,i]/1f3))
    end

    return origins
end

function get_origin_time(X::AbstractArray{Float32}, eikonet, scaler::MinmaxScaler,
                         T_obs::AbstractArray{Float32})
    # # Then determine origin time given hypocenter
    T_pred = Eikonet.solve(X, eikonet, scaler)
    T_pred = dropdims(T_pred, dims=1)
    T_obs = reshape(T_obs, :, 1)
    T_obs = repeat(T_obs, 1, size(X, 3))
    resid = T_obs - T_pred
    origin_offset = median(resid)
    return origin_offset, median(resid, dims=2) .- origin_offset
end

function plot_particles()
    lats = Vector{Float32}()
    lons = Vector{Float32}()    
    for p in eachslice(X, dims=2)
        hypo = inv_trans(ENU(p[1], p[2], 0f0))
        push!(lats, hypo.lat)
        push!(lons, hypo.lon)
    end
    p1 = scatter(lons, lats, xlabel="Longitude", ylabel="Latitude")
    p2 = scatter(lons, -X[3,:]/1f3, xlabel="Longitude", ylabel="Depth")
    p3 = scatter(lats, -X[3,:]/1f3, xlabel="Latitude", ylabel="Depth")
    plot(p1, p2, p3, layout=(3,1))
    savefig("test.png")
end

function format_dtimes(dtimes::DataFrame, stations::DataFrame)
    X_sta = zeros(Float32, nrow(stations), 3)
    phase_sta = innerjoin(dtimes, stations, on = [:network, :station])
    X_inp = zeros(Float32, size(phase_sta, 1), 4)
    X_inp[:,1] .= phase_sta.X
    X_inp[:,2] .= phase_sta.Y
    X_inp[:,3] .= phase_sta.Z
    ΔT_obs = Array{Float32}(phase_sta.dt)
    for (i, row) in enumerate(eachrow(phase_sta))
        if row.phase == "P"
            X_inp[i,4] = 0
        elseif row.phase == "S"
            X_inp[i,4] = 1
        else
            println("Error: unknown Phase label (not P or S). Exiting...")
            println(row)
        end
    end
    return X_inp, ΔT_obs, phase_sta
end

function syntheticdataset(pfile)
    params = JSON.parsefile(pfile)
 
    # Read in a pick file in csv format (e.g. from Gamma or PhaseLink)
    dtimes = CSV.read(params["dtime_file"], DataFrame)

    if params["verbose"]
        println(first(dtimes, 5), "\n")
    end
    dtimes.phase = uppercase.(dtimes.phase) 

    stations = get_stations(params)
    unique!(stations)
    if params["verbose"]
        println(stations)
    end

    active_sta = unique(dtimes[:,[:network, :station]])
    stations = innerjoin(active_sta, stations, on=[:network, :station])
    if nrow(active_sta) != nrow(stations)
        println("Warning: station mismatch between dtime and station files")
        println(nrow(active_sta), " stations in dtime file")
        println(nrow(stations), " stations in station file")
    end
    dtimes = innerjoin(dtimes, stations, on=[:network, :station])[:,names(dtimes)]
    
    eikonet = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    origins = DataFrame(evid=String[], lat=Float32[], lon=Float32[], depth=Float32[])
    for (i, lon) in enumerate(range(-115.75, -115.65, length=100))
        push!(origins, ("$i", 33.3, lon, 5.0))
    end

    trans = ENUfromLLA(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    X, map_3d2d, map_all2row = format_input(stations, origins, trans)
    X = transpose(X)
    X[2:end,:] = forward(X[2:end,:], scaler)

    n_sta = nrow(stations)
    n_events = nrow(origins)
    n_pred = 2 * n_events * n_sta

    phase_label = ["P", "S"]
    I = Vector{Int32}()
    J = Vector{Int32}()
    V = Vector{Float32}()
    count = 1
    n_obs = 2 * n_sta * length(collect(combinations(1:n_events, 2)))
    evids1 = []
    evids2 = []
    nets = []
    stas = []
    phases = []
    for j in 1:n_sta
        for k in 1:2
            for (evid1, evid2) in combinations(origins.evid, 2)
                col1 = map_all2row[(evid1, stations.network[j], stations.station[j], phase_label[k])]
                col2 = map_all2row[(evid2, stations.network[j], stations.station[j], phase_label[k])]
                push!(I, count)
                push!(I, count)
                push!(J, col1)
                push!(J, col2)
                push!(V, -1f0)
                push!(V, 1f0)
                push!(evids1, evid1)
                push!(evids2, evid2)
                push!(nets, stations.network[j])
                push!(stas, stations.station[j])
                push!(phases, phase_label[k])
                count += 1
            end
        end
    end
    Q = sparse(I, J, V, n_obs, n_pred)

    T_pred = dropdims(Eikonet.solve(X[2:end,:], eikonet, scaler), dims=1) .+ X[1,:]
    ΔT_pred = Q*T_pred

    dtimes = DataFrame(dt=ΔT_pred, network=nets, station=stas, evid1=evids1, evid2=evids2, phase=phases)
    println(length(unique(dtimes.evid1)), " ", length(unique(dtimes.evid1)))

    origins0 = DataFrame(time=DateTime[], evid=String[], lat=Float32[], lon=Float32[], depth=Float32[], mag=Float32[])
    for (i, lon) in enumerate(eachrow(origins))
        lat = rand(Normal(33.3, 0.015))
        lon = rand(Normal(-115.7, 0.015))
        timeshift = rand(Normal(0.0, 0.1))
        push!(origins0, (DateTime("1986-11-20T00:00:00")+sec2date(timeshift), "$i", lat, lon, 5.0, NaN32))
    end

    println(first(dtimes, 10))
    println(first(origins, 10))
    println(first(origins0, 10))

    origins = locate(params, origins0, dtimes, stations, eikonet, scaler)

    if params["verbose"]
        println(first(origins, 10))
    end
    CSV.write(params["catalog_outfile"], origins)
    CSV.write(params["catalog_infile"], origins0)
    return
end

function run(pfile)
    params = JSON.parsefile(pfile)
 
    origins0 = CSV.read(params["catalog_infile"], DataFrame, delim=",", types=Dict(:time=>DateTime, :evid=>String, :latitude=>Float32,
        :longitude=>Float32, :depth=>Float32))
    println(first(origins0, 5))
    println()

    # Read in a dtime file in csv format (e.g. Stasis)
    dtimes = CSV.read(params["dtime_file"], DataFrame, types=Dict(:network=>String, :station=>String, #
                                                                  :phase=>String, :cc=>Float32, :dt=>Float32, :evid1=>String,
                                                                  :evid2=>String))#, :diffcc=>Float32))
    

    if params["verbose"]
        println(first(dtimes, 5), "\n")
    end
    dtimes.phase = uppercase.(dtimes.phase) 

    active_evids = DataFrame(evid=unique(cat(dtimes.evid1, dtimes.evid2, dims=1)))
    origins0 = innerjoin(active_evids, origins0, on=[:evid])[:,names(origins0)]
    origins0 = sort(origins0, :evid)
    println(first(origins0, 5))
    println()

    stations = get_stations(params)
    unique!(stations)
    active_sta = unique(dtimes[:,[:network, :station]])
    stations = innerjoin(active_sta, stations, on=[:network, :station])
    if nrow(active_sta) != nrow(stations)
        println("Warning: station mismatch between dtime and station files")
        println(nrow(active_sta), " stations in dtime file")
        println(nrow(stations), " stations in station file")
    end
    if params["verbose"]
        display(stations)
    end
    dtimes = innerjoin(dtimes, stations, on=[:network, :station])[:,names(dtimes)]
    
    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    origins = locate(params, origins0, dtimes, stations, model, scaler)

    if params["verbose"]
        println(first(origins, 10))
    end
    CSV.write(params["catalog_outfile"], origins)

    return
end

function logrange(x1, x2, n)
    (10^y for y in range(log10(x1), log10(x2), length=n))
end

end