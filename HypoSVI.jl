# module HypoSVI

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
using Distributed
using ProgressMeter
using Optim
using LineSearches
using CovarianceEstimation
using Printf

include("./Neuma.jl")
using .Neuma
include("./Adam.jl")
include("./SVIExtras.jl")

abstract type InversionMethod end
abstract type MAP4p <: InversionMethod end
abstract type MAP3p <: InversionMethod end
abstract type SVI <: InversionMethod end

function logit(p::Float32)
    log(p / (1f0-p))
end

function sigmoid(x::Float32)
    1f0 / (1f0 + exp(-x))
end

function setup_priors(params, scaler::MinmaxScaler)
    prior_μ = zeros(Float32, 3)
    prior_μ[1:2] .= 5f-1
    prior_μ[3] = Float32((params["prior_depth_mean"] - scaler.min[3]) / scaler.scale)

    prior_Σ = Float32.(params["prior_cov"] / scaler.scale)
    prior_x = MvNormal(prior_μ, prior_Σ)
    return prior_x
end

function locate(params, X::Array{Float32}, T_obs::Array{Float32}, eikonet::EikoNet, T_ref, ::Type{MAP3p})
    n_phase = size(X, 2)
    ipairs = collect(combinations(collect(1:n_phase), 2))
    ipairs = permutedims(hcat(ipairs...))
    ΔT_obs = T_obs[ipairs[:,1]] - T_obs[ipairs[:,2]]

    scaler = data_scaler(params)
    X = forward(X, scaler)
    X_src = Float32.([0.5, 0.5, 0.5])
    X_rec = X[4:end,:,:]
    # First determine hypocenter with dtimes

    function loss(X_src::AbstractArray)
        # X_trans = [if i == 3 exp(X_src[i]) else X_src[i] end for i in 1:3]
        X_in = cat(repeat(X_src, 1, size(X_rec, 2)), X_rec, dims=1)
        T_pred = dropdims(eikonet(X_in), dims=1)
        ΔT_pred = T_pred[ipairs[:,1]] - T_pred[ipairs[:,2]]
        ℓL = Flux.mae(ΔT_obs, ΔT_pred)
        return ℓL
    end

    lower = Float32.([0.0, 0.0, 0.0])
    upper = Float32.([1.0, 1.0, 1.0])
    result = optimize(
            loss,
            lower,
            upper,
            X_src,
            Fminbox(BFGS(linesearch=LineSearches.BackTracking())),
            Optim.Options(iterations=params["n_epochs"],
            f_tol=params["iter_tol"]),
            autodiff = :forward)
    X_best = vec(Optim.minimizer(result))

    if isnan(X_best[1])
        X_best = fill(5f-1, 3)
    end

    X[1:3,:] .= X_best

    # Then determine origin time given hypocenter
    T_src, resid = get_origin_time(X, eikonet, T_obs)

    X = inverse(X, scaler)

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:3,1,:] .* 1f3

    # Convert X back to meters
    X_best = dropdims(median(X, dims=2), dims=2)

    inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    hypo_lla = inv_trans(ENU(X_best[1], X_best[2], 0f0))

    return Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X_best[3]/1f3, T_ref + sec2date(T_src),
                  NaN, NaN, NaN, NaN, X_best[1]/1f3, X_best[2]/1f3, [], [], [], []), resid
end

function locate(params, X::Array{Float32}, T_obs::Array{Float32}, eikonet::EikoNet, T_ref, ::Type{MAP4p})
    n_phase = size(X, 2)

    scaler = data_scaler(params)
    X = forward(X, scaler)
    X_src = Float32.([0.5, 0.5, 0.5])
    X_rec = X[4:end,:,:]
    θ̂ = [0f0, X_src...]

    # First determine hypocenter with dtimes
    σ = sqrt(2f0) * Float32(params["phase_unc"])

    function loss(θ̂::AbstractArray)
        X_src = θ̂[2:4]
        t0 = θ̂[1]
        X_in = cat(repeat(X_src, 1, size(X_rec, 2)), X_rec, dims=1)
        T_pred = dropdims(eikonet(X_in), dims=1) .+ t0
        ℓL = Flux.huber_loss(vec(T_obs), T_pred, δ=Float32(1.35)*σ)
        return ℓL
    end

    lower = Float32.([-Inf, 0.0, 0.0, 0.0])
    upper = Float32.([Inf, 1.0, 1.0, 1.0])
    hz = Fminbox(BFGS(linesearch=LineSearches.HagerZhang()))
    bt = Fminbox(BFGS(linesearch=LineSearches.BackTracking()))
    options = Optim.Options(iterations=params["n_epochs"], g_tol=params["iter_tol"])
    result = nothing
    try
        result = optimize(loss, lower, upper, θ̂, hz, options, autodiff = :forward)
    catch
        result = optimize(loss, lower, upper, θ̂, bt, options, autodiff = :forward)
    end
    X_best = vec(Optim.minimizer(result))
    X_best = X_best[2:4]

    if isnan(X_best[1])
        X_best = fill(5f-1, 3)
    end

    X[1:3,:] .= X_best

    # Then determine origin time given hypocenter
    T_src, resid = get_origin_time(X, eikonet, T_obs)

    X = inverse(X, scaler)

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:3,1,:] .* 1f3

    # Convert X back to meters
    X_best = dropdims(median(X, dims=2), dims=2)

    inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    hypo_lla = inv_trans(ENU(X_best[1], X_best[2], 0f0))

    return Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X_best[3]/1f3, T_ref + sec2date(T_src),
                  NaN, NaN, NaN, NaN, X_best[1]/1f3, X_best[2]/1f3, [], [], [], []), resid
end

struct HuberDensity{T}
    δ::T
    ε::T
end

function HuberDensity(δ::Float32)
    # source: https://stats.stackexchange.com/questions/210413/generating-random-samples-from-huber-density
    y = 2f0 * pdf(Normal(0f0, 1f0), δ) / δ - 2f0 * cdf(Normal(0f0, 1f0), -δ)
    ε = y / (1+y)
    return HuberDensity(δ, ε)
end

function log_prob(dist::HuberDensity, x::T) where T
    if abs(x) < dist.δ
        ρ = 5.0f-1 * x^2
    else
        ρ = dist.δ * abs(x) - 5.0f-1 * dist.δ^2
    end
    return log((1f0-dist.ε)/sqrt(2f0 * T(π))) - ρ
end

function log_prob(dist::HuberDensity, x::AbstractArray)
    map(x->log_prob(dist, x), x)
end

function RBF_kernel(X::AbstractArray{Float32})
    # This is a specific algorithm for computing pairwise distances fast
    n = size(X, 1)
    G = X * X'
    d² = diag(G) .+ diag(G)' .- 2f0 .* G 
    h = median(d²) / (2.0 * log(n+1))
    γ = 1f0 / (1f-8 + 2 * h)
    K = exp.(-γ * d²)
    return K
end

function RBF_kernel(X::AbstractArray{Float32}, h::Float32)
    # This is a specific algorithm for fast computation of pairwise distances
    n = size(X, 1)
    G = X * X'
    d² = diag(G) .+ diag(G)' .- 2f0 .* G 
    γ = 1f0 / (1f-8 + 2f0 * h)
    K = exp.(-γ * d²)
    return K
end

function median_bw_heuristic(X::AbstractArray{Float32})
    n = size(X, 1)
    G = X * X'
    d² = diag(G) .+ diag(G)' .- 2f0 .* G 
    h = median(d²) / (2f0 * log(n+1f0))
    return h
end

function get_origin_time(X::AbstractArray{Float32}, eikonet::EikoNet, T_obs::AbstractArray{Float32})
    # # Then determine origin time given hypocenter
    T_pred = dropdims(eikonet(X), dims=1)
    T_obs = reshape(T_obs, :, 1)
    T_obs = repeat(T_obs, 1, size(X, 3))
    resid = T_obs - T_pred
    origin_offset = mean(resid)
    return origin_offset, mean(resid, dims=2) .- origin_offset
end

function equal_diff_time(ΔT_pred::AbstractArray, ΔT_obs::AbstractArray, Σ::AbstractArray)
    N_obs = size(ΔT_pred, 1)
    return -N_obs * log(sum(pdf.(Normal.(ΔT_pred, Σ), ΔT_obs)))
end

function logit(x::Float32)
    return log(x / (1.0 - x))
end

function locate(params, X::Array{Float32}, T_obs::Array{Float32}, eikonet::EikoNet, T_ref, ::Type{SVI})
    N = params["n_particles"]
    η = Float32(params["lr"])
    n_phase = size(X, 2)

    ipairs = collect(combinations(collect(1:n_phase), 2))
    ipairs = permutedims(hcat(ipairs...))
    ΔT_obs = T_obs[ipairs[:,1]] - T_obs[ipairs[:,2]]

    X = forward(X, eikonet)
    X_src = Float32.(rand(MvNormal([0.5, 0.5, 0.5], [0.1, 0.1, 0.1]), N))
    X_src = reshape(X_src, 3, 1, N)
    X[1:3,:,:] .= X_src

    X = inverse(X, eikonet)
    X_rec = X[4:end,:,:]
    X_src = mean(X[1:3,:,:], dims=2)

    # X_src = logit.(clamp.(X_src, 0.0, 1.0))

    ΔT_obs = reshape(ΔT_obs, :, 1)
    ΔT_obs = repeat(ΔT_obs, 1, N)
    K = zeros(Float32, N, N)
    ∇K = zeros(Float32, 3, N, N)

    opt = Adam(X_src, η)
    X_last = zeros(Float32, 3, 1, N)

    Σ_pick = params["phase_unc"]
    function ℓπ(X_src::Array)
        # X = cat(repeat(sigmoid(X_src), 1, n_phase, 1), X_rec, dims=1)
        X = cat(repeat(X_src, 1, n_phase, 1), X_rec, dims=1)
        T_pred = dropdims(solve(X, eikonet), dims=1)
        ΔT_pred = T_pred[ipairs[:,1],:] - T_pred[ipairs[:,2],:]
        Σ_tt = clamp.(params["tt_error_fraction"] .* T_pred, params["min_tt_error"], params["max_tt_error"])
        Σ_tot = sqrt.(Σ_tt.^2 .+ Σ_pick^2)
        Σ_pairs = sqrt.(Σ_tot[ipairs[:,1]].^2 + Σ_tot[ipairs[:,2]].^2)
        # ℓL = equal_diff_time(ΔT_pred, ΔT_obs, Σ_pairs)
        ℓL = logpdf.(Laplace.(ΔT_pred, Σ_pairs), ΔT_obs)
        loss = sum(ℓL) #+ sum(ℓ_prior)
        return loss
    end

    L_best = -Inf
    for i in 1:params["n_epochs"]
        
        L, ∇L = Zygote.withgradient(ℓπ, X_src)
        ∇L = ∇L[1]
        # ∇L = ForwardDiff.gradient(ℓπ, X_src)
        # L = ℓπ(X_src)

        ∇L = dropdims(sum(∇L[1:3,:,:], dims=2), dims=2)'

        # X_src_real = inverse(sigmoid(X_src), scaler)
        X_src_real = X_src

        h = median_bw_heuristic(dropdims(X_src_real, dims=2)')
        K = RBF_kernel(dropdims(X_src_real, dims=2)', h)
        ∇RBF(x) = sum(RBF_kernel(x, h))
        ∇K = Zygote.gradient(∇RBF, dropdims(X_src_real, dims=2)')[1]
        ϕ = -1f0 * transpose((K * ∇L .- ∇K) ./ size(K, 1))

        step!(opt, Float32.(Flux.unsqueeze(ϕ, 2)))
        # step!(opt, -∇L)
        X_src = opt.theta

        if L > L_best
            # X[1:3,:,:] .= sigmoid(X_src)
            X[1:3,:,:] .= X_src
            L_best = L
        end

        println("Epoch $i $L $L_best")
    end

    T_src, resid = get_origin_time(X, eikonet, T_obs)

    # Reduce X over arrivals
    X = mean(X, dims=2)[1:3,1,:]

    # Convert X back to meters
    X .*= 1f3
    X_best = dropdims(median(X, dims=2), dims=2)

    # Gaussian approx of posterior uncertainty
    X_cov = cov(BiweightMidcovariance(), X' ./ 1.0f3)
    z_unc = sqrt(X_cov[3,3])
    h_unc = sqrt.(eigvals(X_cov[1:2,1:2]))
    sort!(h_unc)

    inv_trans = LLAfromENU(LLA(lat=params["lat_min"], lon=params["lon_min"]), wgs84)
    hypo_lla = inv_trans(ENU(X_best[1], X_best[2], 0f0))

    if false
        plot_particles(params, X, inv_trans)
    end

    return Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X_best[3]/1f3,
                  T_ref + sec2date(T_src), NaN, h_unc[2], h_unc[1], z_unc, X_best[1]/1f3, X_best[2]/1f3,
                  [], [], [], []), resid
end

function plot_particles(params, X_in)
    X = copy(X_in) / 1.0f3
    X[1:2,:] .-= mean(X[1:2,:], dims=2)
    xlims=(-5.0, 5.0)
    ylims=(-5.0, 5.0)
    zlims=(-params["z_max"], -params["z_min"])
    p1 = scatter(X[1,:], X[2,:], xlabel="Longitude", ylabel="Latitude", xlims=xlims, ylims=ylims)
    p2 = scatter(X[1,:], -X[3,:], xlabel="Longitude", ylabel="Depth", xlims=xlims, ylims=zlims)
    p3 = scatter(X[2,:], -X[3,:], xlabel="Latitude", ylabel="Depth", xlims=ylims, ylims=zlims)
    plot(p1, p2, p3, layout=(3,1), size=(400,800), left_margin = 20Plots.mm)
    savefig("test.png")
end

function plot_particles(params, X, inv_trans)
    lats = Vector{Float32}()
    lons = Vector{Float32}()    
    for p in eachslice(X, dims=2)
        hypo = inv_trans(ENU(p[1], p[2], 0f0))
        push!(lats, hypo.lat)
        push!(lons, hypo.lon)
    end
    xlims=(params["lon_min"], params["lon_max"])
    ylims=(params["lat_min"], params["lat_max"])
    zlims=(-1*params["z_max"], params["z_max"])
    p1 = scatter(lons, lats, xlabel="Longitude", ylabel="Latitude", xlims=xlims, ylims=ylims)
    p2 = scatter(lons, -X[3,:]/1f3, xlabel="Longitude", ylabel="Depth", xlims=xlims, ylims=zlims)
    p3 = scatter(lats, -X[3,:]/1f3, xlabel="Latitude", ylabel="Depth", xlims=ylims, ylims=zlims)
    plot(p1, p2, p3, layout=(3,1), size=(400,800), left_margin = 20Plots.mm)
    savefig("test.png")
end

function update_ssst!(params::Dict, ssst::DataFrame, row::DataFrameRow, max_dist::Float32, kdtree::KDTree,
                 resid::DataFrame, origins::DataFrame)
    sub_ssst = ssst[ssst.evid .== row.evid, :]
    max_kNN = min(params["k-NN"], size(origins, 1))
    idx, dists = knn(kdtree, [row.X, row.Y, row.depth], max_kNN, true)
    idx = idx[dists .<= max_dist]
    local_ssst = Dict()
    for phase in eachrow(sub_ssst)
        local_ssst[(phase.network, phase.station, phase.phase)] = Vector{Float32}()
    end

    for idx in idx
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

function update_alt(params, ssst::DataFrame, row::DataFrameRow, max_dist::Float32, kdtree::KDTree,
                resid::DataFrame, origins::DataFrame)
    max_kNN = min(params["k-NN"], size(origins, 1))
    idx, dists = knn(kdtree, [row.X, row.Y, row.depth], max_kNN, true)

    local_ssst = Dict()
    for phase in eachrow(ssst[ssst.evid .== row.evid, :])
        local_ssst[(phase.network, phase.station, phase.phase)] = Vector{Float32}()
    end

    if length(idx) >= params["min_neighbors"]
        for idx in idx
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
            local_ssst[key] = mean(local_ssst[key])
        else
            local_ssst[key] = 0f0
        end
    end
    return local_ssst
end

function update(params, ssst::DataFrame, row::DataFrameRow, max_dist::Float32, kdtree::KDTree,
                resid::DataFrame, origins::DataFrame)

    if params["k-NN"] >= 1
        max_kNN = min(params["k-NN"], size(origins, 1))
        idx, dists = knn(kdtree, [row.X, row.Y, row.depth], max_kNN)
    else
        idx = inrange(kdtree, [row.X, row.Y, row.depth], max_dist)
    end

    sub_resid = filter(:evid => in(Set(origins.evid[idx])), resid)

    local_ssst = Dict()
    gdf = groupby(sub_resid, [:network, :station, :phase])
    for (key, subdf) in pairs(gdf)
        network, station, phase = values(key)
        if nrow(subdf) >= params["min_neighbors"]
            local_ssst[(network, station, phase)] = median(subdf.residual)
        end
    end

    return local_ssst
end

function update_ssst!(ssst::DataFrame, origins::DataFrame, resid::DataFrame, params::Dict, max_dist::Float32)

    # Now loop over events and find k-Nearest
    kdtree = KDTree(cat(origins.X, origins.Y, origins.depth, dims=2)')

    println("Begin updating SSSTs")
    results = @showprogress @distributed (append!) for row in eachrow(origins)
        local_ssst = update(params, ssst, row, max_dist, kdtree, resid, origins)
        [(row.evid, local_ssst)]
    end

    println("Finished computing SSSTs, updating dataframe")
    @showprogress for (evid, local_ssst) in results
        for phase in eachrow(ssst[ssst.evid .== evid, :])
            key = (phase.network, phase.station, phase.phase)
            if haskey(local_ssst, key)
                ssst.residual[phase.idx] += local_ssst[key]
            end
        end
    end
end

function apply_ssst(phases_old::DataFrame, ssst::DataFrame)
    phases = deepcopy(phases_old)
    sort!(phases, [:evid, :network, :station, :phase])
    sort!(ssst, [:evid, :network, :station, :phase])
    for i in 1:nrow(phases)
        sgn = sign(ssst.residual[i])
        resid = abs(ssst.residual[i])
        sec = Second(floor(resid))
        msec = Millisecond(floor((resid - floor(resid)) * 1000.))
        if sgn >= 0
            phases.time[i] = phases.time[i] - (sec + msec)
        else
            phases.time[i] = phases.time[i] + (sec + msec)
        end
    end
    return phases
end

function init_ssst(phases::DataFrame, resid::DataFrame)
    ssst_dict = Dict()
    for group in groupby(resid, [:network, :station, :phase])
        row = group[1,:]
        ssst_dict[(row.network, row.station, row.phase)] = median(group.residual)
    end

    ssst = DataFrame(evid=phases.evid, network=phases.network, station=phases.station, phase=phases.phase,
                     residual=zeros(Float32, nrow(phases)), count=zeros(Int64, nrow(phases)), idx=1:nrow(phases))

    for row in eachrow(ssst)
        row.residual = ssst_dict[(row.network, row.station, row.phase)]
    end
    return ssst
end

function init_ssst(phases::DataFrame)
    ssst = DataFrame(evid=phases.evid, network=phases.network, station=phases.station, phase=phases.phase,
                     residual=zeros(Float32, nrow(phases)), count=zeros(Int64, nrow(phases)), idx=1:nrow(phases))
    return ssst
end

function plot_events(origins::DataFrame)
    scatter(origins[!,:longitude], origins[!,:latitude], left_margin = 20Plots.mm)
    savefig("events.png")
end

function locate_events(pfile, cat_assoc::DataFrame, phases::DataFrame, stations::DataFrame; outfile=nothing)
    params = JSON.parsefile(pfile)

    if Int(params["n_particles"]) == 1
        method = MAP4p
    else
        method = SVI
    end

    eikonet = BSON.load(params["model_file"], @__MODULE__)[:eikonet]

    # Loop over events
    origin_df = DataFrame(time=DateTime[], evid=Int[], latitude=Float32[], longitude=Float32[], depth=Float32[],
                          magnitude=Float32[], z_unc=Float32[], X=Float32[], Y=Float32[], rmse=Float32[], mae=Float32[])
    resid_df = DataFrame(evid=Int[], network=String[], station=String[], phase=String[], residual=Float32[])

    println("Begin HypoSVI")
    results = @showprogress @distributed (append!) for phase_sub in groupby(phases, :evid)    
        X_inp, T_obs, T_ref, phase_key = format_arrivals(params, DataFrame(phase_sub), stations)
        origin, resid = locate(params, X_inp, T_obs, eikonet, T_ref, method)
        rmse = sqrt(mean(resid.^2))
        mae = mean(abs.(resid))
        mag = filter(row -> row.evid == phase_sub.evid[1], cat_assoc).mag[1]
        origin_df = DataFrame(time=origin.time, evid=phase_sub.evid[1], latitude=origin.lat, longitude=origin.lon,
                            depth=origin.depth, magnitude=mag, z_unc=origin.unc_z, X=origin.X, Y=origin.Y, rmse=rmse, mae=mae)
        resid_df = DataFrame(evid=Int[], network=String[], station=String[], phase=String[], residual=Float32[])
        for (i, row) in enumerate(eachrow(phase_key))
            push!(resid_df, (phase_sub.evid[1], row.network, row.station, row.phase, resid[i]))
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

function logrange(x1, x2, n)
    (10^y for y in range(log10(x1), log10(x2), length=n))
end

function filter_catalog!(params::Dict, cat_assoc::DataFrame)
    filter!(x -> (x.latitude >= params["lat_min_filter"]) & (x.latitude < params["lat_max_filter"]), cat_assoc)
    filter!(x -> (x.longitude >= params["lon_min_filter"]) & (x.longitude < params["lon_max_filter"]), cat_assoc)
end

function remove_outlier_picks!(params, phases, origins)
    resid_df = DataFrame(evid=Int[], network=String[], station=String[], phase=String[], residual=Float32[])
    filter!(x -> abs(x.resid) <= params["max_resid"], phases)
    good_evids = []
    for group in groupby(phases, :evid)
        if nrow(group) >= params["n_det"]
            push!(good_evids, group.evid[1])
        end
    end
    filter!(:evid => in(Set(good_evids)), phases)
    filter!(:evid => in(Set(good_evids)), origins)
    return nothing
end

function read_phases(params, cat_assoc)
    phases = CSV.read(params["phase_file"], DataFrame)
    phases.phase = uppercase.(phases.phase)
    unique!(phases)
    filter!(:evid => in(Set(cat_assoc.evid)), phases)
    return phases
end

function locate_events_ssst(pfile; stop=nothing, start_on_iter=1)
    params = JSON.parsefile(pfile)
    outfile = params["catalog_outfile"]
    resid_file = params["resid_outfile"]

    cat_assoc = CSV.read(params["catalog_infile"], DataFrame)
    if ~isnothing(stop)
        cat_assoc = cat_assoc[1:stop,:]
    end
    if params["verbose"]
        println("Read in ", nrow(cat_assoc), " events")
    end
    filter_catalog!(params, cat_assoc)
    if params["verbose"]
        println("After filtering ", nrow(cat_assoc), " events remaining")
    end
    phases = read_phases(params, cat_assoc)

    stations = get_stations(params)
    unique!(stations)

    if params["max_k-NN_dist"] == params["min_k-NN_dist"]
        ssst_radius = fill(Float32(params["min_k-NN_dist"]), params["n_ssst_iter"])
    else
        ssst_radius = logrange(Float32(params["min_k-NN_dist"]), Float32(params["max_k-NN_dist"]), params["n_ssst_iter"])
        ssst_radius = reverse(collect(ssst_radius))
    end

    # Initial removal of outlier picks
    remove_outlier_picks!(params, phases, cat_assoc)
    phases0 = deepcopy(phases)

    if start_on_iter == 1
        ssst = init_ssst(phases)
    else
        k = start_on_iter - 1
        ssst = CSV.read("$(resid_file)_iter_$(k).csv", DataFrame)
        phases = apply_ssst(phases0, ssst)
    end

    # SSST iterations
    for k in start_on_iter:params["n_ssst_iter"]
        origins, residuals = locate_events(pfile, cat_assoc, phases, stations, outfile="$(outfile)_iter_$(k)")
        println("MAD residual for iter $(k) radius ",  ssst_radius[k], ": ", mad(residuals.residual))
        update_ssst!(ssst, origins, residuals, params, ssst_radius[k])
        phases = apply_ssst(phases0, ssst)
        CSV.write("$(resid_file)_iter_$(k).csv", ssst)
    end

    # Final iteration
    origins, residuals = locate_events(pfile, cat_assoc, phases, stations, outfile="$(outfile)_iter_final")
    println("MAD residual for iter $(params["n_ssst_iter"]) radius ",  ssst_radius[end], ": ", mad(residuals.residual))

end


# end
