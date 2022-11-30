module HypoSVI
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
using ChainRulesCore: ignore_derivatives
using NearestNeighbors
using Distributed
using ProgressMeter

include("./Input.jl")
include("./Eikonet.jl")
include("./Adam.jl")

abstract type InversionMethod end
abstract type MAP <: InversionMethod end
abstract type SVI <: InversionMethod end

function locate(params, X_inp, T_obs, eikonet, scaler, T_ref, ::Type{MAP})
    # Main function to locate events
    lat0 = 0.5(params["lat_min"] + params["lat_max"])
    lon0 = 0.5(params["lon_min"] + params["lon_max"])
    z0 = params["z_min"]
    η = params["lr"]
    n_phase = size(X_inp, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    point_enu = trans(LLA(lat=lat0, lon=lon0))
    ipairs = collect(combinations(collect(1:n_phase), 2))
    ipairs = permutedims(hcat(ipairs...))
    ΔT_obs = T_obs[ipairs[:,1]] - T_obs[ipairs[:,2]]
    X_src = [point_enu.e, point_enu.n, z0*1e3]
    X = cat(repeat(X_src[1:3], 1, n_phase)', X_inp, dims=2)'
    X = forward(X, scaler)

    # First determine hypocenter with dtimes
    for i in 1:params["n_epochs"]
        function loss(X::AbstractArray)      
            T_pred = Eikonet.solve(X, eikonet, scaler)
            ΔT_pred = T_pred[ipairs[:,1]] - T_pred[ipairs[:,2]]
            Flux.mae(ΔT_pred, ΔT_obs, agg=mean)
        end
        LL, ∇LL = withgradient(loss, X)
        ∇LL = ∇LL[1]
        ∇LL = mean(∇LL[1:3,:], dims=2)
        X[1:3,:] .-= η .* ∇LL

        # Clip gradient at region boundary
        X[findall(X[1:3,:] .< 0.0)] .= 0f0
        X[findall(X[1:3,:] .> 1.0)] .= 1f0
    end

    # Then determine origin time given hypocenter
    T_pred = Eikonet.solve(X, eikonet, scaler)
    T_src = median(T_obs - T_pred)

    X = inverse(X, scaler)'
    X_src = X[1,1:3]
    inv_trans = LLAfromENU(origin, wgs84)
    hypo_lla = inv_trans(ENU(X_src[1], X_src[2], 0f0))
    return Origin(hypo_lla.lat, hypo_lla.lon, X_src[3]/1f3, T_ref + sec2date(T_src), NaN)
end

function compute_kernel(X::AbstractArray{Float32})
    n = size(X, 1)
    d² = zeros(Float32, n, n)

    # first compute K
    @inbounds for i in 1:n
        for j in 1:n
            for k in 1:3
                d²[i,j] += (X[i,k] - X[j,k])^2
            end
        end
    end
    if length(d²[d² .> 0.0]) < 1
        h = Inf32
    else
        h = median(d²[d² .> 0.0]) / Float32(log(n))
    end
    K = exp.(-d²/h)

    # # Now compute grad K
    ∇K = zeros(Float32, 3, n, n)
    @inbounds for i in 1:n
        for j in 1:n
            for k in 1:3
                ∇K[k,i,j] = 2f0 * (X[i,k] - X[j,k]) * K[i,j] / h
            end
        end
    end
    return K, ∇K
end

function compute_kernel!(X::AbstractArray{Float32}, K::AbstractArray{Float32}, ∇K::AbstractArray{Float32})
    n = size(X, 1)

    # first compute K
    @inbounds for i in 1:n
        for j in 1:n
            K[i,j] = 0f0
            for k in 1:3
                K[i,j] += (X[i,k] - X[j,k])^2
            end
        end
    end
    if length(K[K .> 0.0]) < 1
        κ = Inf32
    else
        κ = sqrt(median(K[K .> 0.0]) / 2f0)
    end
    K .= exp.(-K ./ κ^2)

    # # Now compute grad K
    @inbounds for i in 1:n
        for j in 1:n
            for k in 1:3
                ∇K[k,i,j] = 2f0 * (X[i,k] - X[j,k]) * K[i,j] / κ
            end
        end
    end
end

function compute_ϕ!(ϕ::AbstractArray{Float32}, ∇LL::AbstractArray{Float32}, K::AbstractArray{Float32}, ∇K::AbstractArray{Float32})
    N = size(K, 1)
    Nfloat = Float32(N)
    @inbounds for j in 1:N
        for l in 1:3
            ϕ[l,j] = 0f0
            for k in 1:N
                ϕ[l,j] += K[k,j] * ∇LL[k,l] + ∇K[l,k,j]
            end
            ϕ[l,j] /= Nfloat
        end
    end
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

function locate(params, X_inp::Array{Float32}, T_obs::Array{Float32}, eikonet, scaler, T_ref, ::Type{SVI})
    # Main function to locate events
    lat0 = 0.5(params["lat_min"] + params["lat_max"])
    lon0 = 0.5(params["lon_min"] + params["lon_max"])
    dlat = 0.05(params["lat_max"] - params["lat_min"])
    dlon = 0.05(params["lon_max"] - params["lon_min"])
    z0 = params["z_max"]
    N = params["n_particles"]
    η = Float32(params["lr"])
    n_phase = size(X_inp, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    X_src = zeros(Float32, 3, n_phase, N)
    for i in 1:N
        lat1 = rand(Uniform(lat0-dlat, lat0+dlat))
        lon1 = rand(Uniform(lon0-dlon, lon0+dlon))
        point_enu = trans(LLA(lat=lat1, lon=lon1))
        X_src[1,:,i] .= point_enu.e
        X_src[2,:,i] .= point_enu.n
        X_src[3,:,i] .= z0*1e3
    end

    ipairs = collect(combinations(collect(1:n_phase), 2))
    ipairs = permutedims(hcat(ipairs...))
    ΔT_obs = T_obs[ipairs[:,1]] - T_obs[ipairs[:,2]]
    ΔT_obs /= Float32(params["phase_unc"])

    X_inp = reshape(X_inp', 4, n_phase, 1)
    X_inp = repeat(X_inp, 1, 1, N)
    X = cat(X_src, X_inp, dims=1)

    for (i, b) in enumerate(eachslice(X, dims=3))
        X[:,:,i] = forward(b, scaler)
    end

    ΔT_obs = reshape(ΔT_obs, :, 1)
    ΔT_obs = repeat(ΔT_obs, 1, N)
    K = zeros(Float32, N, N)
    ∇K = zeros(Float32, 3, N, N)
    ϕ = zeros(Float32, 3, N)

    opt = Adam(mean(X[1:3,:,:], dims=2), η)
    X_last = zeros(Float32, 3, 1, N)
    for i in 1:params["n_epochs"]
        function loss(X::AbstractArray)
            T_pred = Eikonet.solve(X, eikonet, scaler)
            T_pred = dropdims(T_pred, dims=1)
            ΔT_pred = T_pred[ipairs[:,1],:] - T_pred[ipairs[:,2],:]
            ΔT_pred /= Float32(params["phase_unc"])
            loss = Flux.huber_loss(ΔT_pred, ΔT_obs, agg=sum)
            return loss
        end
        ∇LL = gradient(loss, X)[1]
        ∇LL = dropdims(sum(∇LL[1:3,:,:], dims=2), dims=2)'

        X_src = dropdims(mean(X[1:3,:,:], dims=2), dims=2)'
        K, ∇K = compute_kernel(X_src)
        compute_ϕ!(ϕ, ∇LL, K, ∇K)

        step!(opt, Flux.unsqueeze(ϕ, 2), )
        X[1:3,:,:] .= opt.theta

        X[findall(X[1:3,:,:] .< 0.0)] .= 0f0
        X[findall(X[1:3,:,:] .> 1.0)] .= 1f0
        if i == 0
            X_last = opt.theta
            continue
        end
        ℓ² = sqrt.(sum((opt.theta - X_last).^2))
        Δr = ℓ² * scaler.scale / 1f3
        if Δr < params["iter_tol"]
            if params["verbose"]
                println("Early stopping reached at iter $i")
            end
            break
        end
        X_last = opt.theta
    end

    T_src, resid = get_origin_time(X, eikonet, scaler, T_obs)

    for (i, b) in enumerate(eachslice(X, dims=3))
        X[:,:,i] = inverse(b, scaler)
    end

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:3,1,:]
    X_mean = dropdims(mean(X, dims=2), dims=2)

    # Estimate vertical uncertainty
    z_unc = StatsBase.std(X[3,:,:]) / 1f3

    inv_trans = LLAfromENU(origin, wgs84)
    hypo_lla = inv_trans(ENU(X_mean[1], X_mean[2], 0f0))

    if false
        plot_particles()
    end

    return Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X_mean[3]/1f3,
                  T_ref + sec2date(T_src), NaN32, z_unc, X_mean[1]/1f3, X_mean[2]/1f3, []), resid
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

function update!(params, ssst::DataFrame, row::DataFrameRow, max_dist::Float32, kdtree::KDTree,
                 resid::DataFrame, origins::DataFrame)
    sub_ssst = ssst[ssst.evid .== row.evid, :]
    max_kNN = min(params["k-NN"], size(origins, 1))
    idxs, dists = knn(kdtree, [row.X, row.Y, row.depth], max_kNN, true)
    idxs = idxs[dists .<= max_dist]
    local_ssst = Dict()
    for phase in eachrow(sub_ssst)
        local_ssst[(phase.network, phase.station, phase.phase)] = Vector{Float32}()
    end

    for idx in idxs
        sub_resid = resid[resid.evid .== origins.evid[idx], :]
        for phase in eachrow(sub_resid)
            if !haskey(local_ssst, (phase.network, phase.station, phase.phase))
                continue
            end
            push!(local_ssst[(phase.network, phase.station, phase.phase)], phase.residual)
        end
    end
    for key in keys(local_ssst)
        if length(local_ssst[key]) > 0
            local_ssst[key] = median(local_ssst[key])
        else
            local_ssst[key] = 0f0
        end
    end

    for phase in eachrow(sub_ssst)
        #if haskey(local_ssst, (phase.network, phase.station, phase.phase))
        ssst.residual[phase.idx] += local_ssst[(phase.network, phase.station, phase.phase)]
        #end
    end
end

function update(params, ssst::DataFrame, row::DataFrameRow, max_dist::Float32, kdtree::KDTree,
                resid::DataFrame, origins::DataFrame)
    max_kNN = min(params["k-NN"], size(origins, 1))
    idxs, dists = knn(kdtree, [row.X, row.Y, row.depth], max_kNN, true)
    idxs = idxs[dists .<= max_dist]
    local_ssst = Dict()
    for phase in eachrow(ssst[ssst.evid .== row.evid, :])
        local_ssst[(phase.network, phase.station, phase.phase)] = Vector{Float32}()
    end
    if length(idxs) >= params["min_neighbors"]
        for idx in idxs
            sub_resid = resid[resid.evid .== origins.evid[idx], :]
            for phase in eachrow(sub_resid)
                if !haskey(local_ssst, (phase.network, phase.station, phase.phase))
                    continue
                end
                push!(local_ssst[(phase.network, phase.station, phase.phase)], phase.residual)
            end
        end
    end
    for key in keys(local_ssst)
        if length(local_ssst[key]) > params["min_neighbors"]
            local_ssst[key] = median(local_ssst[key])
        else
            local_ssst[key] = 0f0
        end
    end
    return local_ssst
end

function update!(ssst::DataFrame, origins::DataFrame, resid::DataFrame, params::Dict, max_dist::Float32)

    # Now loop over events and find k-Nearest
    println("Building KDTree")
    kdtree = KDTree(cat(origins.X, origins.Y, origins.depth, dims=2)', leafsize=30)
    println("Begin updating SSSTs")
    @showprogress for row in eachrow(origins)
        update!(params, ssst, row, max_dist, kdtree, resid, origins)
    end
end

function update!(ssst::DataFrame, origins::DataFrame, resid::DataFrame, params::Dict, max_dist::Float32)

    # Now loop over events and find k-Nearest
    println("Building KDTree")
    kdtree = KDTree(cat(origins.X, origins.Y, origins.depth, dims=2)', leafsize=30)
    println("Begin updating SSSTs")
    results = @showprogress @distributed (append!) for row in eachrow(origins)
        local_ssst = update(params, ssst, row, max_dist, kdtree, resid, origins)
        [(row.evid, local_ssst)]
    end
    println("Finished computing SSSTs, updating dataframe")
    @showprogress for (evid, local_ssst) in results
        for phase in eachrow(ssst[ssst.evid .== evid, :])
            ssst.residual[phase.idx] += local_ssst[(phase.network, phase.station, phase.phase)]
        end
    end
end

function plot_events(origins::DataFrame)
    scatter(origins[!,:longitude], origins[!,:latitude], left_margin = 20Plots.mm)
    savefig("events.png")
end

function locate_events(pfile; outfile=nothing, phases=nothing)
    params = JSON.parsefile(pfile)

    if params["inversion_method"] isa String
        params["inversion_method"] = eval(Meta.parse(params["inversion_method"]))
    end
 
    # Read in a pick file in csv format (e.g. from Gamma or PhaseLink)
    if isnothing(phases)
        phases = CSV.read(params["phase_file"], DataFrame)
        unique!(phases)
    end
    if params["verbose"]
        println(first(phases, 5), "\n")
    end
    phases.phase = uppercase.(phases.phase) 

    stations = get_stations(params)
    unique!(stations)
    if params["verbose"]
        println(first(stations, 5), "\n")
    end

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    # Loop over events
    origins = DataFrame(time=DateTime[], evid=Int[], latitude=Float32[], longitude=Float32[], depth=Float32[],
                        z_unc=Float32[], X=Float32[], Y=Float32[])
    residuals = DataFrame(evid=Int[], network=String[], station=String[], phase=String[], residual=Float32[])

    for phase_sub in groupby(phases, :evid)
        X_inp, T_obs, T_ref, phase_key = format_arrivals(DataFrame(phase_sub), stations)
        origin, resid = locate(params, X_inp, T_obs, model, scaler, T_ref, params["inversion_method"])

        push!(origins, (origin.time, phase_sub.evid[1], origin.lat, origin.lon, origin.depth,
                        origin.unc_z, origin.X, origin.Y))
        for (i, row) in enumerate(eachrow(phase_key))
            push!(residuals, (phase_sub.evid[1], row.network, row.station, row.phase, resid[i]))
        end
        if params["verbose"]
            println(last(origins, 1))
        end
    end

    if params["verbose"]
        println(first(origins, 10))
    end
    if isnothing(outfile)
        CSV.write(params["catalog_outfile"], origins)
    else
        CSV.write(outfile, origins)
    end
    # plot_events(origins)
    return origins, residuals
end

function locate_events_mp(pfile; outfile=nothing, phases=nothing)
    params = JSON.parsefile(pfile)

    if params["inversion_method"] isa String
        params["inversion_method"] = eval(Meta.parse(params["inversion_method"]))
    end
 
    # Read in a pick file in csv format (e.g. from Gamma or PhaseLink)
    if isnothing(phases)
        phases = CSV.read(params["phase_file"], DataFrame)
        unique!(phases)
    end
    if params["verbose"]
        println(first(phases, 5), "\n")
    end
    phases.phase = uppercase.(phases.phase) 

    cat_assoc = CSV.read(params["catalog_infile"], DataFrame)
    if params["verbose"]
        println("Read in ", nrow(cat_assoc), " events")
        println("Optional spatial filtering of events")
    end
    filter!(x -> (x.lat >= params["lat_min_filter"]) & (x.lat < params["lat_max_filter"]), cat_assoc)
    filter!(x -> (x.lon >= params["lon_min_filter"]) & (x.lon < params["lon_max_filter"]), cat_assoc)
    filter!(:evid => in(Set(cat_assoc.evid)), phases)
    if params["verbose"]
        println("After filtering ", nrow(cat_assoc), " events remaining")
    end
    
    stations = get_stations(params)
    unique!(stations)
    if params["verbose"]
        println(first(stations, 5), "\n")
    end

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    # Loop over events
    origin_df = DataFrame(time=DateTime[], evid=Int[], latitude=Float32[], longitude=Float32[], depth=Float32[],
                        z_unc=Float32[], X=Float32[], Y=Float32[])
    resid_df = DataFrame(evid=Int[], network=String[], station=String[], phase=String[], residual=Float32[])

    results = @showprogress @distributed (append!) for phase_sub in groupby(phases, :evid)
        X_inp, T_obs, T_ref, phase_key = format_arrivals(DataFrame(phase_sub), stations)
        origin, resid = locate(params, X_inp, T_obs, model, scaler, T_ref, params["inversion_method"])

        origin_df = DataFrame(time=origin.time, evid=phase_sub.evid[1], latitude=origin.lat, longitude=origin.lon,
                              depth=origin.depth, z_unc=origin.unc_z, X=origin.X, Y=origin.Y)
        resid_df = DataFrame(evid=Int[], network=String[], station=String[], phase=String[], residual=Float32[])
        for (i, row) in enumerate(eachrow(phase_key))
            push!(resid_df, (phase_sub.evid[1], row.network, row.station, row.phase, resid[i]))
        end
        if params["verbose"]
            println(origin)
        end
        [(origin_df, resid_df)]
    end

    for (local_origin_df, local_resid_df) in results
        append!(origin_df, local_origin_df)
        append!(resid_df, local_resid_df)
    end

    if params["verbose"]
        println(first(origin_df, 10))
    end
    if isnothing(outfile)
        CSV.write(params["catalog_outfile"], origin_df)
    else
        CSV.write(outfile, origin_df)
    end

    return origin_df, resid_df
end

function update!(phases::DataFrame, ssst::DataFrame)
    sort!(phases, [:evid, :network, :station, :phase])
    sort!(ssst, [:evid, :network, :station, :phase])
    for i in 1:nrow(phases)
        sgn = sign(ssst.residual[i])
        resid = abs(ssst.residual[i])
        sec = Second(floor(resid))
        msec = Millisecond(floor((resid - floor(resid)) * 1000.))
        if sgn > 0
            phases.time[i] = phases.time[i] - (sec + msec)
        else
            phases.time[i] = phases.time[i] + (sec + msec)
        end
    end
end

function init_ssst(phases::DataFrame)
    ssst = DataFrame(evid=phases.evid, network=phases.network, station=phases.station, phase=phases.phase,
                     residual=zeros(Float32, nrow(phases)), count=zeros(Int64, nrow(phases)), idx=1:nrow(phases))
    return ssst
end

function logrange(x1, x2, n)
    (10^y for y in range(log10(x1), log10(x2), length=n))
end

function locate_events_ssst(pfile)
    params = JSON.parsefile(pfile)
    outfile = params["catalog_outfile"]
    resid_file = params["resid_outfile"]

    # Read in a pick file in csv format (e.g. from Gamma or PhaseLink)
    phases = CSV.read(params["phase_file"], DataFrame)
    phases.phase = uppercase.(phases.phase)
    unique!(phases)

    cat_assoc = CSV.read(params["catalog_infile"], DataFrame)
    if params["verbose"]
        println("Read in ", nrow(cat_assoc), " events")
        println("Optional spatial filtering of events")
    end
    filter!(x -> (x.lat >= params["lat_min_filter"]) & (x.lat < params["lat_max_filter"]), cat_assoc)
    filter!(x -> (x.lon >= params["lon_min_filter"]) & (x.lon < params["lon_max_filter"]), cat_assoc)
    filter!(:evid => in(Set(cat_assoc.evid)), phases)
    if params["verbose"]
        println("After filtering ", nrow(cat_assoc), " events remaining")
    end

    stations = get_stations(params)
    unique!(stations)

    ssst = init_ssst(phases)
    ssst_radius = collect(logrange(Float32(params["min_k-NN_dist"]), Float32(params["max_k-NN_dist"]),
                                   params["n_ssst_iter"]))
    ssst_radius = reverse(ssst_radius)
    for k in 1:params["n_ssst_iter"]
        origins, residuals = locate_events_mp(pfile, outfile="$(outfile)_iter_$(k)", phases=phases)
        println("Median residual for iter $(k) radius ",  ssst_radius[k], ": ", median(abs.(residuals.residual)))
        update!(ssst, origins, residuals, params, ssst_radius[k])
        phases = CSV.read(params["phase_file"], DataFrame)
        filter!(:evid => in(Set(cat_assoc.evid)), phases)
        update!(phases, ssst)
        ssst_radius[k]
        CSV.write("$(resid_file)_iter_$(k).csv", ssst)
    end
end

end