module Neuma
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
using Clustering
using Optim
using Random
using Distributed
using ProgressMeter

include("./Input.jl")
include("./Eikonet.jl")
include("./Adam.jl")

abstract type InversionMethod end
abstract type EM <: InversionMethod end
abstract type SGD <: InversionMethod end

function get_origin_time(X::AbstractArray{Float32}, eikonet::Flux.Chain, scaler::MinmaxScaler,
                         T_obs::AbstractArray{Float32})
    ###########
    # Note this function is different from the one in HypoSVI
    ###########
    T_pred = Eikonet.solve(X, eikonet, scaler)
    T_pred = dropdims(T_pred, dims=1)
    T_obs = reshape(T_obs, :, 1)
    T_obs = repeat(T_obs, 1, size(X, 3))
    resid = T_obs - T_pred
    origin_offset = median(resid, dims=1)
    return origin_offset
end

function huber(x::Float32, δ::Float32)
    x_abs = abs(x)
    if x_abs <= δ
        return 0.5x^2
    else
        return δ*(x_abs - 0.5δ)
    end
end

function huber(x, δ::Float32)
    x_abs = abs(x)
    if x_abs <= δ
        return 0.5x^2
    else
        return δ*(x_abs - 0.5δ)
    end
end

function huber(x::AbstractArray, δ::Float32)
    map(x0->huber(x0, δ), x)
end

function huber_pdf(x::Float32)
    # Assumes δ = 1
    huber_pdf(x, 1f0)
end

function huber_pdf(x::Float32, δ::Float32)
    # NOT VALIDATED. UNKNOWN SOURCE
    Φ = cdf(Normal(0f0, 1f0), δ)
    ϕ = pdf(Normal(0f0, 1f0), δ)
    exp(-huber(x, δ)) / (2f0 * sqrt(2f0 * π) * (Φ - ϕ/δ - 5f-1))
end

function plot_events(origins::DataFrame)
    scatter(origins[!,:lon], origins[!,:lat], left_margin = 20Plots.mm)
    origins = CSV.read("/scratch/zross/oak_ridge/scsn_cat.csv", DataFrame)
    scatter!(origins[!,:lon], origins[!,:lat], left_margin = 20Plots.mm)
    savefig("events.png")
end

function trace_const_vel(X_t::Array{Float32}, scaler::MinmaxScaler)
    X = cat(map(x->inverse(x, scaler), eachslice(X_t, dims=3))..., dims=3)
    vel = [Float32(6000.0), Float32(6000.0/sqrt(3))]
    v = vel[1 .+ Int.(X[7,:,:])]
    T_pred = sqrt.((X[1,:,:] - X[4,:,:]).^2 + (X[2,:,:] - X[5,:,:]).^2 + (X[3,:,:] - X[6,:,:]).^2) ./ v
    return T_pred
end

function trace_const_vel(X_t::Array, scaler::MinmaxScaler)
    X = cat(map(x->inverse(x, scaler), eachslice(X_t, dims=3))..., dims=3)
    vel = [Float32(6000.0), Float32(6000.0/sqrt(3))]
    v = vel[1 .+ Int.(X[7,:,:])]
    T_pred = sqrt.((X[1,:,:] - X[4,:,:]).^2 + (X[2,:,:] - X[5,:,:]).^2 + (X[3,:,:] - X[6,:,:]).^2) ./ v
    return T_pred
end

function init_X(params::Dict, X_phase::Array{Float32}, t_min::Float32, t_max::Float32)
    K = params["n_clusters"]
    n_obs = size(X_phase, 1)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)

    X_src = zeros(Float32, 4, n_obs, K)
    for i in 1:K
        lat1 = Float32(0.5(params["lat_min"]+params["lat_max"]))
        lon1 = Float32(0.5(params["lon_min"]+params["lon_max"]))
        # z1 = Float32(0.)
        z1 = Float32(0.25(params["z_min"]+params["z_max"]))
        point_enu = trans(LLA(lat=lat1, lon=lon1))
        X_src[2,:,i] .= point_enu.e
        X_src[3,:,i] .= point_enu.n
        X_src[4,:,i] .= z1*1f3
    end
    if K > 1
        X_src[1,:,:] .= reshape(collect(range(t_min, t_max, length=K)), 1, K)
    end

    X_phase = reshape(X_phase', 4, n_obs, 1)
    X_phase = repeat(X_phase, 1, 1, K)
    X = cat(X_src, X_phase, dims=1)
    return X
end

function locate_events(params::Dict, X::Array{Float32}, T_obs::Array{Float32}, γ::Array{Float32},
                       scaler::MinmaxScaler, eikonet::Flux.Chain; iter_tol=0f0)
    # Loop for one M-step
    z_min_tr = (0f0 - scaler.min)/scaler.scale
    z_max_tr = (params["z_max"]*1f3 - scaler.min)/scaler.scale
    η = Float32(params["lr"])
    X_best = X_src = mean(X[1:4,:,:], dims=2)
    ℓ_best = Inf
    X_rec = X[5:end,:,:]
    opt = Adam(X_src, η)
    ℓ_last = Inf
    idx_best = Inf

    function loss(X_src::AbstractArray)
        X_in = cat(repeat(X_src, 1, size(X_rec, 2), 1), X_rec, dims=1)
        T_pred = dropdims(Eikonet.solve(X_in[2:end,:,:], eikonet, scaler), dims=1) .+ X_in[1,:,:]
        #Q_θ = mean(huber(T_pred .- T_obs', Float32(params["huber_delta"])) .* γ)
        Q_θ = mean(sum(huber(T_pred .- T_obs', Float32(params["huber_delta"])) .* γ, dims=2))
        # Q_θ = mean((T_pred .- T_obs').^2 .* γ)
        # Q_θ = mean(abs.(T_pred .- T_obs') .* γ)
        return Q_θ
    end

    for i in 1:params["n_epochs"]
        ℓ, ∇Q_θ = withgradient(loss, X_src)
        ∇Q_θ = ∇Q_θ[1]
        # X_src = X_src - η*∇Q_θ
        # X_src[2:3,:,:] = clamp.(X_src[2:3,:,:], 0f0, 1f0)
        # X_src[4,:,:] = clamp.(X_src[4,:,:], z_min_tr, z_max_tr)
        step!(opt, ∇Q_θ)
        opt.theta[2:3,:,:] = clamp.(opt.theta[2:3,:,:], 0f0, 1f0)
        opt.theta[4,:,:] = clamp.(opt.theta[4,:,:], z_min_tr, z_max_tr)
        X_src = opt.theta
        Δℓ = ℓ-ℓ_best
        if ℓ < ℓ_best
            X_best = X_src
            ℓ_best = ℓ
            idx_best = i
        end
    end
    return X_best, ℓ_best
end

function assign_γ(T_pred::Array{Float32}, T_obs::Array{Float32})
    γ = zeros(Float32, size(T_pred))
    resid = abs.(T_pred .- T_obs')
    idx = argmin(resid, dims=2)
    γ[idx] .= 1f0
    return γ
end

function assign_γ(T_pred::Array{Float32}, T_obs::Array{Float32}, ϕ::Array{Float32}, dist::Distribution)
    pr = pdf.(dist, T_pred .- T_obs') .* reshape(ϕ, 1, :)  .+ 1f-8
    denom = sum(pr, dims=2)
    γ = pr ./ denom
    return γ
end

function GMM(params, X_phase::Array{Float32}, T_obs::Array{Float32}, amp_obs::Array{Float32}, eikonet::Flux.Chain,
             scaler::MinmaxScaler, T_ref, picks::DataFrame)

    X = init_X(params, X_phase, minimum(T_obs), maximum(T_obs))
    for (i, slice) in enumerate(eachslice(X[2:end,:,:], dims=3))
        X[2:end,:,i] = forward(slice, scaler)
    end

    K = params["n_clusters"]
    K_plus = K+1
    n_obs = size(X_phase, 1)
    G_t_dist = Float32(1.0 / (maximum(T_obs) - minimum(T_obs)))
    G_amp_dist = MixtureModel(Normal[
        Normal(-6f0, 1f0),
        Normal(-5.1f0, 1f0)])
    ϕ = zeros(Float32, K_plus)
    ϕ[1:K] .= 1f0
    ϕ ./= sum(ϕ)
    γ_best = γ = zeros(Float32, n_obs, K_plus) # Doesn't need to be initialized
    i_best = 0
    X_best = X
    M = zeros(Float32, n_obs, K)
    M_best = M
    n_decay = params["n_warmup_iter"]
    phase_unc = collect(1*range(params["phase_unc"], 5.0*params["phase_unc"], length=n_decay))
    phase_unc = reverse(convert(Vector{Float32}, phase_unc))
    amp_unc = collect(1*range(params["amp_unc"], 5.0*params["amp_unc"], length=n_decay))
    amp_unc = reverse(convert(Vector{Float32}, amp_unc))

    ℓ_best = -Inf32
    ℓ_last = ℓ_best
    hypo_dist = sqrt.((X[2,:,:]-X[5,:,:]).^2 .+ (X[3,:,:]-X[6,:,:]).^2 .+ (X[4,:,:]-X[7,:,:]).^2)
    hypo_dist .*= scaler.scale / 1f3
    M = compute_mag.(hypo_dist, amp_obs')

    for iter in 1:params["n_warmup_iter"]
        L_t_unc = Laplace(Float32(0.), phase_unc[iter])
        L_amp_unc = Laplace(Float32(0.), amp_unc[iter])
        # L_t_unc = Normal(Float32(0.), phase_unc[iter])
        # L_amp_unc = Normal(Float32(0.), amp_unc[iter])
        T_pred = dropdims(Eikonet.solve(X[2:end,:,:], eikonet, scaler), dims=1) + X[1,:,:]
        amp_pred = GMPE.(hypo_dist, M)

        pr_eq = pdf.(L_amp_unc, amp_pred .- amp_obs') .* pdf.(L_t_unc, T_pred .- T_obs') .* ϕ[1:K]'
        ϕ[K_plus] = 0f0 # forcing ϕ_G to be zero during warmup
        pr_G = ϕ[K_plus] .* Float32.(pdf.(G_amp_dist, amp_obs)) .* G_t_dist
        pr = cat(pr_eq, pr_G', dims=2) .+ 1f-8
        ℓ = sum(log.(sum(pr, dims=2)))
        γ = pr ./ sum(pr, dims=2)

        X[1:4,:,:] .= locate_events(params, X, T_obs, γ[:,1:K], scaler, eikonet, iter_tol=params["iter_tol"])[1]
        hypo_dist = sqrt.((X[2,:,:]-X[5,:,:]).^2 .+ (X[3,:,:]-X[6,:,:]).^2 .+ (X[4,:,:]-X[7,:,:]).^2)
        hypo_dist .*= scaler.scale / 1f3
        M = compute_mag.(hypo_dist, amp_obs')

        # if ℓ > ℓ_best
        ℓ_best = ℓ
        X_best .= X
        M_best .= M
        γ_best .= γ
        i_best = iter
        # end

        ℓ_last = ℓ
        Nₖ = sum(γ, dims=1)
        ϕ .= vec(Nₖ / n_obs)
        if params["verbose"]
            println("warmup iter $iter $ℓ $ℓ_best ", vec(Nₖ))
        end
    end

    X = X_best
    γ = γ_best
    Nₖ = sum(γ, dims=1)
    ϕ .= vec(Nₖ) ./ n_obs
    ϕ[K_plus] = 1f-2
    ϕ ./= sum(ϕ)

    ℓ_best = -Inf32
    ℓ_last = ℓ_best
    hypo_dist = sqrt.((X[2,:,:]-X[5,:,:]).^2 .+ (X[3,:,:]-X[6,:,:]).^2 .+ (X[4,:,:]-X[7,:,:]).^2)
    hypo_dist .*= scaler.scale / 1f3
    M = compute_mag.(hypo_dist, amp_obs')

    for iter in 1:params["n_iter"]
        L_t_unc = Laplace(Float32(0.), phase_unc[end])
        L_amp_unc = Laplace(Float32(0.), amp_unc[end])
        # L_t_unc = Normal(Float32(0.), phase_unc[end])
        # L_amp_unc = Normal(Float32(0.), amp_unc[end])
        T_pred = dropdims(Eikonet.solve(X[2:end,:,:], eikonet, scaler), dims=1) + X[1,:,:]
        amp_pred = GMPE.(hypo_dist, M)

        pr_eq = pdf.(L_amp_unc, amp_pred .- amp_obs') .* pdf.(L_t_unc, T_pred .- T_obs') .* ϕ[1:K]'
        pr_G = ϕ[K_plus] .* Float32.(pdf.(G_amp_dist, amp_obs)) .* G_t_dist
        pr = cat(pr_eq, pr_G', dims=2) .+ 1f-8
        ℓ = sum(log.(sum(pr, dims=2)))
        γ = pr ./ sum(pr, dims=2)

        X[1:4,:,:] .= locate_events(params, X, T_obs, γ[:,1:K], scaler, eikonet, iter_tol=params["iter_tol"])[1]
        hypo_dist = sqrt.((X[2,:,:]-X[5,:,:]).^2 .+ (X[3,:,:]-X[6,:,:]).^2 .+ (X[4,:,:]-X[7,:,:]).^2)
        hypo_dist .*= scaler.scale / 1f3
        M = compute_mag.(hypo_dist, amp_obs')

        if ℓ > ℓ_best
            ℓ_best = ℓ
            X_best .= X
            M_best .= M
            γ_best .= γ
            i_best = iter
        end

        ℓ_last = ℓ
        Nₖ = sum(γ, dims=1)
        ϕ .= vec(Nₖ / n_obs)
        if params["verbose"]
            println("full iter $iter $ℓ $ℓ_best ", vec(Nₖ))
        end
    end

    X .= X_best
    γ .= γ_best
    Nₖ = sum(γ, dims=1)
    ϕ .= vec(Nₖ / n_obs)
    hypo_dist = sqrt.((X[2,:,:]-X[5,:,:]).^2 .+ (X[3,:,:]-X[6,:,:]).^2 .+ (X[4,:,:]-X[7,:,:]).^2)
    hypo_dist .*= scaler.scale / 1f3
    M = compute_mag.(hypo_dist, amp_obs')

    for (i, slice) in enumerate(eachslice(X[2:end,:,:], dims=3))
        X[2:end,:,i] = inverse(slice, scaler)
    end

    γ = argmax(γ, dims=2)
    γ = vec([x[2] for x in γ])
    X = median(X, dims=2)[1:4,1,:]

    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    inv_trans = LLAfromENU(origin, wgs84)
    results = Vector{Origin}()

    for i in sort(unique(γ))
        if i == K_plus
            continue
        end
        arrivals = picks[findall(γ .== i), [:arid, :network, :station, :phase, :time]]
        mag = median(M[findall(γ .== i), i])
        if nrow(arrivals) < params["n_det"]
            continue
        end
        hypo_lla = inv_trans(ENU(X[2,i], X[3,i], 0f0))
        hypo = Origin(Float32(hypo_lla.lat), Float32(hypo_lla.lon), X[4,i]/1f3,
                      T_ref + sec2date(X[1,i]), mag, NaN, X[2,i]/1f3, X[3,i]/1f3, arrivals.arid)
        push!(results, hypo)
    end
    return results
end

function evaluate_results(pred_arrivals, true_arrivals)
    # First do Jaccard precision
    J_p = []
    total = []
    for group in groupby(pred_arrivals, :evid)
        push!(J_p, [])
        #total = length(group.arid)
        push!(total, length(group.arid))
        for group2 in groupby(true_arrivals, :evid)
            common = length(intersect(group.arid, group2.arid))
            push!(J_p[end], common)
        end
    end
    J_p = sum([maximum(x) for x in J_p]) / sum(total)

    J_r = []
    total = []
    for group in groupby(true_arrivals, :evid)
        if group.evid == 0
            continue
        end
        push!(J_r, [])
        #total = length(group.arid)
        push!(total, length(group.arid))
        for group2 in groupby(pred_arrivals, :evid)
            common = length(intersect(group.arid, group2.arid))
            push!(J_r[end], common)
        end
    end
    J_r = sum([maximum(x) for x in J_r]) / sum(total)
    return J_p, J_r
end

function plot_clusters(params, X, T, amp, γ, γ_true)
    l = @layout [a b; c d]
    size = 5.0*(amp .- minimum(amp) .+ 2.0) ./ (maximum(amp) - minimum(amp))
    p1 = scatter(X[5,:]/1000., T, color=:lightgrey, markersize=size)
    p2 = scatter(X[6,:]/1000., T, color=:lightgrey, markersize=size)
    counter = 1
    for idx in sort(unique(γ))
        idx2 = findall(γ .== idx)
        if length(idx2) < params["n_det"]
            continue
        end
        if idx > params["n_clusters"]
            scatter!(p1, X[5,idx2]/1000., T[idx2], label="Garbage", color=:white, markersize=size[idx2])
            scatter!(p2, X[6,idx2]/1000., T[idx2], label="Garbage", color=:white, markersize=size[idx2])
        else
            scatter!(p1, X[5,idx2]/1000., T[idx2], label=counter, markersize=size[idx2])
            scatter!(p2, X[6,idx2]/1000., T[idx2], label=counter, markersize=size[idx2])
        end
        counter += 1
    end
    plot(p1, p2, layout = l, size=(700, 900))
    savefig("test.png")
end

function softmax(f)
    return exp.(f) / sum(exp.(f))
end

function GMPE(R::Float32, M::Float32)
    # Takes in R in km
    # Outputs PGV in cm/s
    c0, c1, c2, c3 = 1.08f0, 0.93f0, -0.015f0, -1.68f0
    log_PGV = c0 + c1*(M-3.5f0) + c3*log10(max(R, 0.1f0))
    return log_PGV
end

function compute_mag(R::Float32, log_PGV::Float32)
    # Takes in R in km, PGV in cm/s
    c0, c1, c2, c3 = 1.08f0, 0.93f0, -0.015f0, -1.68f0
    M = (log_PGV - c0 - c3*log10(max(R, 0.1f0)))/c1 + 3.5f0
    return M
end

function generate_syn_dataset(params::Dict, stations::DataFrame, eikonet::Flux.Chain, scaler::MinmaxScaler; n_events=3, t_max=6f1, n_fake=100, seed=1234, max_picks_per_event=Inf)
    rng = MersenneTwister(seed)
    n_events = Int(n_events)
    T0 = DateTime("1986-11-20T00:00:00.0")
    phases = DataFrame(arid=Int[], network=String[], station=String[], phase=String[], time=DateTime[], amp=Float32[])
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    arid_idx = 1
    event_idx = Vector{Int}()
    offset = rand(rng, Uniform(Float32(15.), Float32(t_max-30.)), Int(n_events))
    # if n_events > 1
    #     offset = collect(range(Float32(15.), t_max-Float32(30.), length=n_events))
    # elseif n_events == 1
    #     offset = [t_max/2.0]
    # end
    mags = Float32.(rand(rng, Uniform(2.8f0, 3f0), n_events))
    amp_dist = MixtureModel(Normal[
        Normal(-6f0, 1f0),
        Normal(-5.1f0, 1f0)])

    phase_unc = Float32(params["phase_unc"])
    amp_unc = Float32(params["amp_unc"])
    L_t_unc = Normal(Float32(0.), phase_unc)
    L_amp_unc = Normal(Float32(0.), amp_unc)

    for i in 1:n_events
        origin_time = T0 + sec2date(offset[i])
        input = zeros(Float32, 7, 1)
        lat1 = Float32.(rand(rng, Uniform(params["lat_min"], params["lat_max"])))
        lon1 = Float32.(rand(rng, Uniform(params["lon_min"], params["lon_max"])))
        z1 = Float32.(rand(rng, Uniform(0.0, 50.0)))
        point_enu = trans(LLA(lat=Float32(lat1), lon=Float32(lon1)))
        input[1:3] = forward_point([point_enu.e, point_enu.n, z1*1f3], scaler)
        temp_phases = []
        temp_idx = []
        dists = []
        for row in eachrow(stations)
            input[4] = row.X / scaler.scale
            input[5] = row.Y / scaler.scale
            input[6] = row.Z / scaler.scale
            hypo_dist = sqrt((point_enu.e-row.X)^2 + (point_enu.n-row.Y)^2 + (z1-row.Z)^2)
            hypo_dist = Float32.(hypo_dist / 1000.0)
            amp = GMPE(hypo_dist, mags[i]) + rand(rng, L_amp_unc)
            for phase_label in ["P", "S"]
                if phase_label == "P"
                    input[7] = 0f0
                else
                    input[7] = 1f0
                end
                T_pred = Eikonet.solve(input, eikonet, scaler)[1] + rand(rng, L_t_unc)
                arrival_time = origin_time + sec2date(T_pred)
                push!(temp_phases, (arid_idx, row.network, row.station, phase_label, arrival_time, amp))
                push!(temp_idx, i)
                push!(dists, hypo_dist)
                arid_idx += 1
            end
        end
        for i in axes(temp_phases, 1)
            push!(phases, temp_phases[i])
            push!(event_idx, temp_idx[i])
        end
    end

    for i in 1:n_fake
        phase_label = rand(rng, ["P", "S"])
        idx = rand(rng, 1:nrow(stations))
        net = stations.network[idx]
        sta = stations.station[idx]
        amp = rand(rng, amp_dist)
        arrival_time = T0 + sec2date(rand(rng, Uniform(0f0, t_max)))
        push!(phases, (arid_idx, net, sta, phase_label, arrival_time, amp))
        push!(event_idx, 0)
        arid_idx += 1
    end
    return phases, event_idx
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

function sliding_picks(t, t_step)
    return (findall((t .>= t0) .&& (t .< (t0+t_step))) for t0 in range(t[1], t[end], step=t_step))
end

function partition_picks(params, T_rel)
    clust = []
    one_day = 86400.0
    for t_start in range(minimum(T_rel), maximum(T_rel), step=one_day)
        t_stop = t_start + one_day + 30.0
        idx = findall((T_rel .>= t_start) .& (T_rel .< t_stop))
        if length(idx) < params["n_det"]
            continue
        end
        T_rel_sub = reshape(T_rel[idx], 1, :)
        local_clusters = dbscan(T_rel_sub, 5.0, min_neighbors=params["dbscan_min_neighbors"], min_cluster_size=params["n_det"], leafsize=params["leafsize"])
        for cluster in local_clusters
            push!(clust, idx[cluster.core_indices])
        end
        println("Day ", t_start/86400.0, " ", length(local_clusters))
    end
    return clust
end

function detect_events(pfile, infile, phase_ofile, cat_ofile)
    params = JSON.parsefile(pfile)

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    stations = get_stations(params)
    phases = CSV.read(infile, DataFrame)

    sncls = map(x->split(x, "."), phases.id)
    insertcols!(phases, :network => [sncl[1] for sncl in sncls])
    insertcols!(phases, :station => [sncl[2] for sncl in sncls])
    rename!(phases, :event_idx => :evid)
    rename!(phases, :timestamp => :time)
    rename!(phases, :type => :phase)
    phases.phase = map(uppercase, phases.phase)
    phases.amp = map(log10, phases.amp.*1f2)
    sort!(phases, [:time])
    insertcols!(phases, :arid => 1:nrow(phases))
    println(first(phases, 10))

    # Loop over events
    origins = DataFrame(time=DateTime[], evid=Int[], lat=Float32[], lon=Float32[], depth=Float32[], mag=Float32[],
                        X=Float32[], Y=Float32[])
    assoc = DataFrame(arid=Int[], network=String[], station=String[], phase=String[], time=DateTime[], evid=Int[])

    T_rel = map(x -> (x.time - phases[1, "time"]).value, eachrow(phases)) ./ 1000.0
    clusters = partition_picks(params, T_rel)
    # T_rel = reshape(T_rel, 1, :)
    # clusters = dbscan(T_rel, 5.0, min_neighbors=params["dbscan_min_neighbors"], min_cluster_size=params["n_det"], leafsize=params["leafsize"])
    println("Begin association with $(length(clusters)) clusters")
    #dur = []
    #for cluster in clusters
    #    idx = cluster.core_indices
    #    phase_sub = phases[idx, :]
    #    #push!(dur, maximum(T_rel[idx])-minimum(T_rel[idx]))
    #    push!(dur, max(4*Int(floor(nrow(phase_sub) / (nrow(stations)))), 4))
    #end
    #sort!(dur)
    #for d in dur
    #    println(d)
    #end

    function parallel_detect(idx)
        if length(idx) < params["n_det"]
            return []
        end
        phase_sub = phases[idx, :]
        params["n_clusters"] = max(4*Int(floor(nrow(phase_sub) / (nrow(stations)))), 4)
        X_phase, T_obs, amp_obs, T_ref, picks = format_arrivals(DataFrame(phase_sub), stations)
        origins = GMM(params, X_phase, T_obs, amp_obs, model, scaler, T_ref, picks)
        for origin in origins
            println(origin.time, " ", origin.lat, " ", origin.lon, " ", origin.depth, " ", origin.mag)
        end
        return origins
    end
    @time all_results = pmap(parallel_detect, clusters)
    # all_results = progress_map(clusters) do x
    #     parallel_detect(x)
    # end
    println("Finished running Neuma, preparing assoc output")
    evid = 1000000
    sort!(phases, :arid)
    for results in all_results
        for hypo in results
            for i in hypo.arids
                push!(assoc, (phases[i,:arid], phases[i,:network], phases[i,:station],
                              phases[i,:phase], phases[i,:time], evid))
            end
            push!(origins, (hypo.time, evid, hypo.lat, hypo.lon, hypo.depth, hypo.mag, hypo.X, hypo.Y))
            evid += 1
        end
    end

    if params["verbose"]
        println(first(origins, 100))
    end
    CSV.write(phase_ofile, assoc)
    CSV.write(cat_ofile, origins)

    return origins
end


function exp_ridgecrest(infile, phase_ofile, cat_ofile)
    # params = JSON.parsefile(pfile)
    params = build_neuma_data_params()

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    stations = get_stations(params)
    phases = CSV.read(infile, DataFrame)
    sncls = map(x->split(x, "."), phases.id)
    insertcols!(phases, :network => [sncl[1] for sncl in sncls])
    insertcols!(phases, :station => [sncl[2] for sncl in sncls])
    rename!(phases, :event_idx => :evid)
    rename!(phases, :timestamp => :time)
    rename!(phases, :type => :phase)
    phases.phase = map(uppercase, phases.phase)
    phases.amp = map(log10, phases.amp.*1f2)
    sort!(phases, [:time])
    insertcols!(phases, :arid => 1:nrow(phases))
    println(first(phases, 10))

    # Loop over events
    origins = DataFrame(time=DateTime[], evid=Int[], lat=Float32[], lon=Float32[], depth=Float32[], mag=Float32[],
                        X=Float32[], Y=Float32[])
    assoc = DataFrame(arid=Int[], network=String[], station=String[], phase=String[], time=DateTime[], evid=Int[])

    T_rel = map(x -> (x.time - phases[1, "time"]).value, eachrow(phases)) ./ 1000.0
    T_rel = reshape(T_rel, 1, :)

    clusters = dbscan(T_rel, 4.0, min_neighbors=1, min_cluster_size=params["n_det"])
    # println("Begin association with $(length(clusters)) clusters")
    # dur = []
    # for cluster in clusters
    #     idx = cluster.core_indices
    #     phase_sub = phases[idx, :]
    #     #push!(dur, maximum(T_rel[idx])-minimum(T_rel[idx]))
    #     push!(dur, max(4*Int(floor(nrow(phase_sub) / (nrow(stations)))), 4))
    # end
    # sort!(dur)
    # for d in dur
    #     println(d)
    # end
    # return

    function parallel_detect(cluster)
        idx = cluster.core_indices
        phase_sub = phases[idx, :]
        params["n_clusters"] = max(4*Int(floor(nrow(phase_sub) / (nrow(stations)))), 4)
        X_phase, T_obs, amp_obs, T_ref, picks = format_arrivals(DataFrame(phase_sub), stations)
        origins = GMM(params, X_phase, T_obs, amp_obs, model, scaler, T_ref, picks)
        for origin in origins
            println(origin.time, " ", origin.lat, " ", origin.lon, " ", origin.depth, " ", origin.mag)
        end
        return origins
    end
    @time all_results = pmap(parallel_detect, clusters)
    # all_results = progress_map(clusters) do x
    #     parallel_detect(x)
    # end
    println("Finished running Neuma, preparing assoc output")
    evid = 1000000
    sort!(phases, :arid)
    for results in all_results
        for hypo in results
            for i in hypo.arids
                push!(assoc, (phases[i,:arid], phases[i,:network], phases[i,:station],
                              phases[i,:phase], phases[i,:time], evid))
            end
            push!(origins, (hypo.time, evid, hypo.lat, hypo.lon, hypo.depth, hypo.mag, hypo.X, hypo.Y))
            evid += 1
        end
    end

    if params["verbose"]
        println(first(origins, 100))
    end
    CSV.write(phase_ofile, assoc)
    CSV.write(cat_ofile, origins)

    return origins
end

function detect_syn_events(infile, pfile, phase_ofile, cat_ofile)
    # params = JSON.parsefile(pfile)
    params = build_neuma_syn_params()

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    stations = get_stations(params)
    phases = CSV.read(infile, DataFrame)

    # Loop over events
    origins = DataFrame(time=DateTime[], evid=Int[], lat=Float32[], lon=Float32[], depth=Float32[], mag=Float32[],
                        X=Float32[], Y=Float32[])
    assoc = DataFrame(arid=Int[], network=String[], station=String[], phase=String[], time=DateTime[], evid=Int[])

    sort!(phases, [:time])
    T_rel = map(x -> (x.time - phases[1, "time"]).value, eachrow(phases)) ./ 1000.0
    T_rel = reshape(T_rel, 1, :)

    clusters = dbscan(T_rel, 1.0, min_neighbors=1, min_cluster_size=params["n_det"])
    #     dur = []
    # for cluster in clusters
    #     idx = cluster.core_indices
    #     phase_sub = phases[idx, :]
    #     #push!(dur, maximum(T_rel[idx])-minimum(T_rel[idx]))
    #     push!(dur, max(4*Int(floor(nrow(phase_sub) / (nrow(stations)))), 4))
    # end
    # sort!(dur)
    # for d in dur
    #     println(d)
    # end
    # return
    println("Begin association with $(length(clusters)) clusters")

    function parallel_detect(cluster)
        idx = cluster.core_indices
        phase_sub = phases[idx, :]
        params["n_clusters"] = max(4*Int(floor(nrow(phase_sub) / (nrow(stations)))), 4)
        X_phase, T_obs, amp_obs, T_ref, picks = format_arrivals(DataFrame(phase_sub), stations)
        origins = GMM(params, X_phase, T_obs, amp_obs, model, scaler, T_ref, picks)
        for origin in origins
            println(origin.time, " ", origin.lat, " ", origin.lon, " ", origin.depth, " ", origin.mag)
        end
        return origins
    end
    @time all_results = pmap(parallel_detect, clusters)
 
    evid = 1000000
    for results in all_results
        for i in eachindex(results)
            hypo, arrivals = results[i]
            insertcols!(arrivals, :evid => fill(evid, nrow(arrivals)))
            assoc = vcat(assoc, arrivals)
            push!(origins, (hypo.time, evid, hypo.lat, hypo.lon, hypo.depth, hypo.mag, hypo.X, hypo.Y))
            evid += 1
        end
    end

    J_p, J_r = evaluate_results(assoc, phases[phases.evid .> 0, :])
    println("Super γ Jp: $J_p")
    println("Super γ Jr: $J_r")

    assoc = CSV.read("gamma_D2.csv", DataFrame)
    insertcols!(assoc, :arid => phases.arid)
    rename!(assoc,:event_index => :evid)
    filter!(row -> row.evid > 0, assoc)
    J_p, J_r = evaluate_results(assoc, phases[phases.evid .> 0, :])
    println("γ Jp: $J_p")
    println("γ Jr: $J_r")

    assoc = CSV.read("phaselink_picks.csv", DataFrame)
    J_p, J_r = evaluate_results(assoc, phases[phases.evid .> 0, :])
    println("PhaseLink Jp: $J_p")
    println("PhaseLink Jr: $J_r")

    if params["verbose"]
        println(first(origins, 100))
    end
    # plot_events(origins)
    CSV.write(phase_ofile, assoc)
    CSV.write(cat_ofile, origins)

    return origins
end

end
