module Neuma
__precompile__

using DataFrames
using CSV
using JSON
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
using Peaks
using KernelDensity
# using Clustering
using Random
using Distributed
using ProgressMeter
using Printf
using Optim
using LineSearches
using SpecialFunctions

include("./Input.jl")
include("./Eikonet.jl")
include("./NeumaParams.jl")
include("./NeumaPlot.jl")
include("./NeumaOptim.jl")
# include("./NeumaCRP.jl")
include("./NeumaPaperSyn.jl")

function hypo_dist(X::AbstractArray)
    if size(X, 1) == 8
        R = sqrt.((X[2,:,:]-X[5,:,:]).^2 .+ (X[3,:,:]-X[6,:,:]).^2 .+ (X[4,:,:]-X[7,:,:]).^2)
    elseif size(X, 1) == 7
        R = sqrt.((X[1,:,:]-X[4,:,:]).^2 .+ (X[2,:,:]-X[5,:,:]).^2 .+ (X[3,:,:]-X[6,:,:]).^2)        
    end
    return R
end

function logspace(start, stop, length)
    10 .^ (range(start,stop=stop,length=length))
end

function unique_sta_phase(X::Array{Float32})
    X_rec = X[5:end,:,1]
    n_obs = size(X_rec, 2)
    u_rows = unique(X_rec, dims=2)
    labels = Vector{Int}()
    for i in 1:n_obs
        idx = argmin(dropdims(sum(abs.(u_rows .- X_rec[i]), dims=2), dims=2))
        push!(labels, idx)
    end
    return labels
end

function softmax(f)
    return exp.(f) / sum(exp.(f))
end

function GMPE(R::Real, M::Real)
    # Takes in R in km
    # Outputs PGV in cm/s
    c0, c1, c2 = 1.08f0, 0.93f0, -1.68f0
    log_PGV = c0 + c1*(M-3.5f0) + c2*log10(max(R, 0.1f0))
    # log_PGV = c3*log10(max(R, 0.1f0))
    return log_PGV
end

function compute_mag(R::Real, log_PGV::Real)
    # Takes in R in km, PGV in cm/s
    c0, c1, c2, c3 = 1.08f0, 0.93f0, -0.015f0, -1.68f0
    M = (log_PGV - c0 - c3*log10(max(R, 0.1f0)))/c1 + 3.5f0
    return M
end

function build_dataset_only(fname; seed=1, n_events=1)
    params = build_neuma_syn_params()

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    stations = get_stations(params)
    phases, event_idx = generate_syn_dataset(params, stations, model, scaler;
        n_events=n_events, n_fake=57600, seed=seed, max_picks_per_event=Inf, t_max=86400.0)
    insertcols!(phases, :evid => event_idx)
    CSV.write(fname, phases)
end

function extract_detections(params, result, T_ref, sub_phase_sta)
    X, M, γ, γ_ij, resid = result
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    inv_trans = LLAfromENU(origin, wgs84)
    results = Vector{Origin}()
    K = size(X)[end]

    if params["verbose"]
        @printf("%d picks total in cluster\n", size(γ, 1))
        println([nrow(sub_phase_sta[findall(γ .== i),:]) for i in 1:K])
    end

    for i in sort(unique(γ))
        idx = findall(γ .== i)
        if ndims(resid) == 2
            event_resid = resid[idx, i]
            mag_resid = rand(5)
        else
            event_resid = resid[idx, 1, i]
            mag_resid = resid[idx, 2, i]
        end
        arrivals = sub_phase_sta[idx, [:arid, :network, :station, :phase, :time]]
        mag = median(M[idx, i])
        event_γ = γ_ij[idx, i]
        unique_phase = unique(arrivals[:, [:network, :station, :phase]])
        if params["count_unique_only"]
            if nrow(unique_phase) < params["n_det"]
                continue
            end
        else
            if nrow(arrivals) < params["n_det"]
                continue
            end
        end
        hypo_lla = inv_trans(ENU(X[2,i], X[3,i], 0f0))
        origin_time = T_ref + sec2date(X[1,i])
        hypo = Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X[4,i]/1f3, origin_time, mag, 0f0, 0f0, 0f0,
                       X[2,i]/1f3, X[3,i]/1f3, arrivals.arid, event_resid, mag_resid, event_γ)
        push!(results, hypo)
    end
    otimes = [result.time for result in results]
    idx = sortperm(otimes)
    results = results[idx]
    return results
end

function init_X(params::Dict, X_phase::Array{Float32}, t_min::Float32, t_max::Float32; rand_init=false)
    rng = MersenneTwister(1234)
    K = params["n_clusters"]
    n_obs = size(X_phase, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    X_src = zeros(Float32, 4, n_obs, K)
    for i in 1:K
        if rand_init
            lat1 = rand(rng, Uniform(params["lat_min"], params["lat_max"]))
            lon1 = rand(rng, Uniform(params["lon_min"], params["lon_max"]))
            z1 = rand(rng, Uniform(params["z_min"], params["z_max"]))
        else
            lat1 = Float32(0.5(params["lat_min"]+params["lat_max"]))
            lon1 = Float32(0.5(params["lon_min"]+params["lon_max"]))
            z1 = Float32(params["z_max"] + params["z_min"]) * 5f-1
        end
        point_enu = trans(LLA(lat=lat1, lon=lon1))
        X_src[2,:,i] .= point_enu.e
        X_src[3,:,i] .= point_enu.n
        X_src[4,:,i] .= z1*1f3
    end

    if K > 1
        if rand_init
            X_src[1,:,:] .= rand(rng, Uniform(t_min, t_max), 1, K)
        else
            X_src[1,:,:] .= reshape(collect(range(t_min, t_max, length=K)), 1, K)
        end
    else
        X_src[1,:,:] .= t_min
    end

    X_phase = reshape(X_phase', 4, n_obs, 1)
    X_phase = repeat(X_phase, 1, 1, K)
    X = cat(X_src, X_phase, dims=1)

    X[2:7,:,:] ./= 1f3

    return X
end

function KD_clustering(params, x_obs::Vector{T}) where T
    x = range(x_obs[1], x_obs[end]+T(1.0), step=T(0.5))
    U = kde(x_obs, x; bandwidth = params["kernel_bw"])
    p = U.density
    x = U.x

    clusters = []
    i_last = 1
    for i in argminima(p)
        x_max = x[i]
        x_min = x[i_last]
        i_min = searchsortedfirst(x_obs, x_min)
        i_max = searchsortedlast(x_obs, x_max)
        i_last = i
        if (i_max - i_min) >= params["n_det"]
            push!(clusters, collect(i_min:i_max))
        end
    end
    x_min = x[i_last]
    i_min = searchsortedfirst(x_obs, x_min)
    push!(clusters, collect(i_min:length(x_obs)))
    return clusters
end

function initial_clusters(params, T_rel)
    clusters = KD_clustering(params, T_rel)
    # clusters = dbscan(T_rel, params["dbscan_eps"], min_neighbors=params["dbscan_min_neighbors"], min_cluster_size=params["n_det"])
    # labels = clusters.assignments
    # cluster_idx = unique(labels)
    # cluster_idx = cluster_idx[cluster_idx .> 0]

    for cluster in clusters
        idx = cluster
        window_dur = maximum(T_rel[idx]) - minimum(T_rel[idx])
        if window_dur >= 360.0
            @warn("Found DBSCAN cluster with duration $window_dur sec. Performance may be degraded")
        end
    end

    if length(params["restrict_clusters_to"]) > 0
        clusters = [clusters[x] for x in params["restrict_clusters_to"]]
    end

    return clusters
end

function prepare_inputs(params, process_year)
    eikonet = BSON.load(params["model_file"], @__MODULE__)[:eikonet]
    display(eikonet)

    stations = get_stations(params)
    if length(params["restrict_coords_to"]) > 0
        filter!(row -> row.latitude >= params["restrict_coords_to"][3], stations)
        filter!(row -> row.latitude < params["restrict_coords_to"][4], stations)
        filter!(row -> row.longitude >= params["restrict_coords_to"][1], stations)
        filter!(row -> row.longitude < params["restrict_coords_to"][2], stations)
    end
    if ~params["read_subset_only"]
        phases = CSV.read(params["phase_infile"], DataFrame)
    else
        phases = CSV.read(params["phase_infile"], DataFrame, limit=10000)
    end
    filter!(row -> year(row.time) == Int(process_year), phases)
    @printf("%d phases read in initially\n", nrow(phases))
    phases = innerjoin(phases, stations, on = [:network, :station])[:, names(phases)]
    @printf("%d phases remaining after removing those not in station list\n", nrow(phases))

    phases.phase = map(uppercase, phases.phase)
    sort!(phases, [:time])
    insertcols!(phases, :arid => 1:nrow(phases))
    println(first(phases, 10))

    T_rel = sort(map(x -> (x.time - phases[1, "time"]).value, eachrow(phases)) ./ 1000.0)

    clusters = initial_clusters(params, T_rel)
    return eikonet, phases, stations, T_rel, clusters
end

function build_output_dataframes(all_results, phases::DataFrame, evid)
    origins = DataFrame(time=DateTime[], evid=Int[], latitude=Float32[], longitude=Float32[],
                        depth=Float32[], mag=Float32[], X=Float32[], Y=Float32[])
    assoc = DataFrame(arid=Int[], network=String[], station=String[], phase=String[],
                      time=DateTime[], evid=Int[], resid=Float32[], prob=Float32[])
    for results in all_results
        for hypo in results
            for (i, arid) in enumerate(hypo.arids)
                push!(assoc, (arid, phases[arid,:network], phases[arid,:station], phases[arid,:phase],
                              phases[arid,:time], evid, hypo.resid[i], hypo.prob[i]))
            end
            push!(origins, (hypo.time, evid, hypo.lat, hypo.lon, hypo.depth, hypo.mag, hypo.X, hypo.Y))
            evid += 1
        end
    end
    sort!(origins, [:time])
    return origins, assoc
end

function detect(params, idx::Array{Int}, T_rel::Array{Float64}, phases::DataFrame, stations::DataFrame, eikonet::EikoNet)
    if length(idx) < params["n_det"]
        return Vector{Origin}()
    end
    sub_phase = phases[idx, :]

    window_dur = maximum(T_rel[idx]) - minimum(T_rel[idx])
    params["n_clusters"] = Int(round(window_dur/params["min_interevent_time"])) + 2
    
    if params["verbose"]
        @printf("Processing cluster of %d picks\n", nrow(sub_phase))
    end
    X, y_obs, T_ref, sub_phase_sta = format_arrivals(params, DataFrame(sub_phase), stations)
    results = associate(params, X, y_obs, eikonet)
    if isnothing(results)
        return Vector{Origin}()
    end
    origins = extract_detections(params, results, T_ref, sub_phase_sta)
    for origin in origins
        if params["verbose"]
            @printf("%s %.4f %.4f %.2f %.2f %d %.2f %.2f %.2f\n",
                origin.time, origin.lat, origin.lon, origin.depth, origin.mag, length(origin.resid), sqrt(mean(origin.resid.^2)),
                mean(abs.(origin.resid)), sqrt(mean(origin.mags.^2)))
        end
    end
    if params["verbose"]
        println()
    end
    return origins
end

function ridgecrest_scsn_exp(pfile; run_pmap=true)
    params = JSON.parsefile(pfile)

    eikonet, phases, stations, T_rel, clusters = prepare_inputs(params)

    println("Init Neuma with ", length(clusters), " clusters")

    if run_pmap
        f = progress_pmap
    else
        f = progress_map
    end

    results = f(clusters) do cluster
        detect(params, cluster, T_rel, phases, stations, eikonet)
    end

    println("Finished running Neuma, preparing assoc output")
    origins, assoc = build_output_dataframes(results, phases)

    println(first(origins, 100))

    CSV.write(@sprintf("%s_phase_iter_%d_phaseunc_%.02f_ampunc_%.02f_mdist_%s_edist_%s.csv",
                       params["label"], params["EM_epochs"], 
                       params["phase_unc"], params["amp_unc"], params["mstep_dist"], params["estep_dist"]), assoc)
    CSV.write(@sprintf("%s_cat_iter_%d_phaseunc_%.02f_ampunc_%.02f_mdist_%s_edist_%s.csv",
                       params["label"], params["EM_epochs"], 
                       params["phase_unc"], params["amp_unc"], params["mstep_dist"], params["estep_dist"]), origins)
    return origins
end

function merge_catalogs(pfile)
    params = JSON.parsefile(pfile)
    assoc = []
    origins = []
    evid_max = 1000000
    for year in params["years_to_process"]
        try
            assoc_ = CSV.read(@sprintf("%s_%s", params["phase_outfile"], year), DataFrame)
            origins_ = CSV.read(@sprintf("%s_%s", params["catalog_outfile"], year), DataFrame)
            evid_min = minimum(origins_.evid)
            assoc_.evid .+= evid_max - evid_min
            origins_.evid .+= evid_max - evid_min
            evid_max = maximum(origins_.evid) + 1
            push!(assoc, assoc_)
            push!(origins, origins_)
        catch
            continue
        end
    end
    assoc = vcat(assoc...)
    origins = vcat(origins...)
    CSV.write(@sprintf("%s", params["phase_outfile"]), assoc)
    CSV.write(@sprintf("%s", params["catalog_outfile"]), origins)
end

function run(pfile; run_pmap=true)
    params = JSON.parsefile(pfile)
    evid = 1000000
    for year in params["years_to_process"]
        println("Now processing year $year")
        eikonet, phases, stations, T_rel, clusters = prepare_inputs(params, year)

        println("Using ", nrow(stations), " stations.")
        println("Init Neuma with ", length(clusters), " clusters")

        f = if run_pmap progress_pmap else progress_map end

        results = f(clusters) do cluster
            detect(params, cluster, T_rel, phases, stations, eikonet)
        end

        println("Finished running Neuma, preparing assoc output")
        origins, assoc = build_output_dataframes(results, phases, evid)

        println(first(origins, 10))

        CSV.write(@sprintf("%s_%s", params["phase_outfile"], year), assoc)
        CSV.write(@sprintf("%s_%s", params["catalog_outfile"], year), origins)
    end
end

end
