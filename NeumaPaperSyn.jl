using Glob
using Dates
using CSV
using DataFrames
using Printf
using JSON
using Geodesy
using ProgressMeter
using SharedArrays
# include("./Neuma.jl")
# include("./Input.jl")

function sec_to_date(s::Union{Float32, Float64})
    sec_sign = Int32(sign(s))
    s = abs(s)
    sec = Int32(floor(s))
    msec = Int32(floor(1000*(s - sec)))
    return Dates.Second(sec * sec_sign) + Dates.Millisecond(msec * sec_sign)
end

function GMPE(R::Real, M::Real)
    # Takes in R in km
    # Outputs PGV in cm/s
    c0, c1, c2 = 1.08f0, 0.93f0, -1.68f0
    log_PGV = c0 + c1*(M-3.5f0) + c2*log10(max(R, 0.1f0))
    # log_PGV = c3*log10(max(R, 0.1f0))
    return log_PGV
end

function read_SCSN_phase(fname::String, max_station_dist::Float64)
    picks = Dict()
    if ~isfile(fname)
        return picks
    end
    open(fname) do file
        lines = readlines(file)
        origin_time = nothing
        evid = nothing
        mag = nothing
        depth = nothing
        for i in 1:length(lines)
            temp = split(lines[i])
            if i == 1
                evid = temp[1]
                temp2 = split(temp[4], ",")
                ymd = temp2[1]
                hms = temp2[2]
                year, month, day = split(ymd, "/")
                hour, minute, second = split(hms, ":")
                year = parse(Int32, year)
                month = parse(Int32, month)
                day = parse(Int32, day)
                hour = parse(Int32, hour)
                minute = parse(Int32, minute)
                sec = Int32(floor(parse(Float32, second)))
                msec = Int32(floor(1000*(parse(Float32, second) - sec)))
                origin_time = DateTime(
                    year, month, day, hour, minute, sec, msec)
                mag = parse(Float32, temp[8])
                depth = parse(Float32, temp[7])
            else
                net = temp[1]
                sta = temp[2]
                phase = temp[8]
                dist = parse(Float64, temp[length(temp)-1])
                if dist >= max_station_dist
                    continue
                end
                R = sqrt(dist^2 + depth^2)
                amp = GMPE(R, mag)
                pick = parse(Float32, temp[end])
                picks[(evid, net, sta, phase)] = (origin_time + sec_to_date(pick), amp)
            end
        end
    end
    return picks
end

function main()
    # function generate_ridgecrest_dataset()
    files = Glob.glob("phase_files/*.phase")
    phase_picks = Dict()
    for file in files
        println(file, " ", length(phase_picks))
        # phase_picks += read_SCSN_phase(file, 10000.0)
        new_picks = read_SCSN_phase(file, 10000.0)
        phase_picks = merge(phase_picks, new_picks)
    end
    # end

    phase = DataFrame(network=String[], station=String[], phase=String[], time=DateTime[], evid=String[], amp=Float32[])
    for key in keys(phase_picks)
        # println(key)
        push!(phase, (key[2], key[3], key[4], phase_picks[key][1], key[1], phase_picks[key][2]))
    end

    CSV.write("SCSN_ridgecrest_true_seq.csv", phase)
end

# main()

function evaluate_results2(pred_arrivals::DataFrame, true_arrivals::DataFrame)
    # First do Jaccard precision
    out = @showprogress @distributed (+) for group in groupby(pred_arrivals, :evid)
        J_p_local = []
        total = length(group.arid)
        for group2 in groupby(true_arrivals, :evid)
            common = length(intersect(group.arid, group2.arid))
            push!(J_p_local, common)
        end
        [[maximum(J_p_local), total]]
    end
    J_p = [x[1] for x in out]
    total = [x[2] for x in out]
    println("Jp ", sum(J_p), " ", sum(total))
    J_p = sum(J_p) / sum(total)

    out = @showprogress @distributed (+) for group in groupby(true_arrivals, :evid)
        J_r_local = []
        total = length(group.arid)
        for group2 in groupby(pred_arrivals, :evid)
            common = length(intersect(group.arid, group2.arid))
            push!(J_r_local, common)
        end
        [[maximum(J_r_local), total]]
    end
    J_r = [x[1] for x in out]
    total = [x[2] for x in out]
    println("Jr ", sum(J_r), " ", sum(total))
    J_r = sum(J_r) / sum(total)
    return J_p, J_r
end

function phase_jaccard(pred_arrivals::DataFrame, true_arrivals::DataFrame)
    # First do Jaccard precision

    J_p = @showprogress @distributed (append!) for group in groupby(pred_arrivals, :evid)
        J_p_local = []
        for group2 in groupby(true_arrivals, :evid)
            push!(J_p_local, length(intersect(group.arid, group2.arid)) / length(union(group.arid, group2.arid)))
            # push!(J_p_local, length(intersect(group.arid, group2.arid)) / nrow(group))
            # push!(J_p_local, length(intersect(group.arid, group2.arid)))
        end
        [maximum(J_p_local)]
    end
    J_p = mean(J_p)
    # J_p = sum(J_p) / nrow(pred_arrivals)

    J_r = @showprogress @distributed (append!) for group in groupby(true_arrivals, :evid)
        J_r_local = []
        for group2 in groupby(pred_arrivals, :evid)
            push!(J_r_local, length(intersect(group.arid, group2.arid)) / length(union(group.arid, group2.arid)))
            # push!(J_r_local, length(intersect(group.arid, group2.arid)) / nrow(group))
            # push!(J_r_local, length(intersect(group.arid, group2.arid)))
        end
        [maximum(J_r_local)]
    end
    J_r = mean(J_r)
    # J_r = sum(J_r) / nrow(true_arrivals)
    return J_p, J_r
end

function event_jaccard(pred_arrivals::DataFrame, true_arrivals::DataFrame)
    # First do Jaccard precision
    out = @showprogress @distributed (append!) for group in groupby(pred_arrivals, :evid)
        J_p_local = []
        for group2 in groupby(true_arrivals, :evid)
            push!(J_p_local, length(intersect(group.arid, group2.arid)) / length(union(group.arid, group2.arid)))
        end
        if maximum(J_p_local) >= 0.5
            score = 1.0
        else
            score = 0.0
        end
        [score]
    end
    J_p = sum(out) / length(out)

    out = @showprogress @distributed (append!) for group in groupby(true_arrivals, :evid)
        J_r_local = []
        for group2 in groupby(pred_arrivals, :evid)
            push!(J_r_local, length(intersect(group.arid, group2.arid)) / length(union(group.arid, group2.arid)))
        end
        if maximum(J_r_local) >= 0.5
            score = 1.0
        else
            score = 0.0
        end
        [score]
    end
    J_r = sum(out) / length(out)
    return J_p, J_r
end

function pairwise_metrics(pred_arrivals::DataFrame, true_arrivals::DataFrame, n_phase_min)
    δ_pred = Set()
    δ_true = Set()

    for group in groupby(pred_arrivals, :evid)
        if nrow(group) >= n_phase_min
            for pair in combinations(group.arid, 2)
                a, b = sort(pair)
                push!(δ_pred, (a, b))
            end
        end
    end
    for group in groupby(true_arrivals, :evid)
        if group.evid[1] < 1.0
            continue
        end
        for pair in combinations(group.arid, 2)
            a, b = sort(pair)
            push!(δ_true, (a, b))
        end
    end

    TP = 0
    for pair in δ_pred
        if pair in δ_true
            TP += 1
        end
    end
    precision = TP / length(δ_pred)
    recall = TP / length(δ_true)
    return precision, recall
end


function evaluate_neuma(pfile)
    params = JSON.parsefile(pfile)

    stations = get_stations(params)
    if ~params["read_subset_only"]
        phases = CSV.read(params["phase_infile"], DataFrame)
    else
        phases = CSV.read(params["phase_infile"], DataFrame, limit=10000)
    end
    @printf("%d phases read in initially\n", nrow(phases))
    phases = innerjoin(phases, stations, on = [:network, :station])[:, names(phases)]
    @printf("%d phases remaining after removing those not in station list\n", nrow(phases))

    phases.phase = map(uppercase, phases.phase)
    sort!(phases, [:time])
    insertcols!(phases, :arid => 1:nrow(phases))
    println(first(phases, 10))


    assoc = CSV.read(@sprintf("%s_phase_iter_%d_phaseunc_%.02f_ampunc_%.02f_mdist_%s_edist_%s.csv",
                                params["label"], params["EM_epochs"], 
                                params["phase_unc"], params["amp_unc"], params["mstep_dist"], params["estep_dist"]), DataFrame)
    println(first(assoc))
    sort!(assoc, [:time])
    # rename!(assoc,:event_index => :evid)
    filter!(row -> row.evid > 0, assoc)
    for k in 8:8#10
        J_p, J_r = pairwise_metrics(assoc, phases, k)
        println("Neuma pairwise $k $J_p $J_r")
    end


    # J_p, J_r = event_jaccard(assoc, phases)
    # println("Neuma Jp: $J_p")
    # println("Neuma Jr: $J_r")

    # J_p, J_r = phase_jaccard(assoc, phases)
    # println("Neuma Jp: $J_p")
    # println("Neuma Jr: $J_r")
end


function evaluate_gamma(pfile)
    params = JSON.parsefile(pfile)

    stations = get_stations(params)
    if ~params["read_subset_only"]
        phases = CSV.read(params["phase_infile"], DataFrame)
    else
        phases = CSV.read(params["phase_infile"], DataFrame, limit=10000)
    end
    @printf("%d phases read in initially\n", nrow(phases))
    phases = innerjoin(phases, stations, on = [:network, :station])[:, names(phases)]
    @printf("%d phases remaining after removing those not in station list\n", nrow(phases))

    phases.phase = map(uppercase, phases.phase)
    sort!(phases, [:time])
    insertcols!(phases, :arid => 1:nrow(phases))
    println(first(phases, 10))

    assoc = CSV.read("picks_gamma_fake_d2.csv", DataFrame)
    println(first(assoc))
    sort!(assoc, [:phase_time])
    insertcols!(assoc, :arid => phases.arid)
    rename!(assoc,:event_index => :evid)
    filter!(row -> row.evid > 0, assoc)
    for k in 8:8
        J_p, J_r = pairwise_metrics(assoc, phases, k)
        println("γ $k $J_p $J_r")
    end
end

function generate_scsn_plus_fake_dataset(pfile, outfile, n_fake)
    rng = MersenneTwister(1234)
    params = JSON.parsefile(pfile)

    stations = get_stations(params)
    if ~params["read_subset_only"]
        phases = CSV.read(params["phase_infile"], DataFrame)
    else
        phases = CSV.read(params["phase_infile"], DataFrame, limit=10000)
    end
    @printf("%d phases read in initially\n", nrow(phases))
    phases = innerjoin(phases, stations, on = [:network, :station])[:, names(phases)]
    @printf("%d phases remaining after removing those not in station list\n", nrow(phases))
    phases.phase = map(uppercase, phases.phase)
    sort!(phases, [:time])
    # insertcols!(phases, :arid => 1:nrow(phases))
    # select!(phases, Not(:arid))
    println(first(phases, 10))


    for i in 1:nrow(phases)
        phases.amp[i] += rand(rng, Normal(0f0, 0.25))
    end

    amp_dist = Normal(-5.1e0, 2e0)

    t_max = maximum(map(x -> (x.time - phases[1, "time"]).value, eachrow(phases)) ./ 1000.0)

    for i in 1:n_fake
        phase_label = rand(rng, ["P", "S"])
        idx = rand(rng, 1:nrow(stations))
        net = stations.network[idx]
        sta = stations.station[idx]
        amp = rand(rng, amp_dist)
        arrival_time = phases.time[1] + sec2date(rand(rng, Uniform(0e0, t_max)))
        push!(phases, (net, sta, phase_label, arrival_time, 0, amp))
    end

    CSV.write(outfile, phases)
end

function generate_syn_dataset(params::Dict, stations::DataFrame, eikonet::Flux.Chain, scaler::MinmaxScaler;
    n_events=3, t_max=6e1, n_fake=100, seed=1234, max_picks_per_event=Inf, use_amp=true,
    max_dist=100.0)
    rng = MersenneTwister(seed)
    n_events = Int(n_events)
    T0 = DateTime("1986-11-20T00:00:00.0")
    if use_amp
        phases = DataFrame(arid=Int[], network=String[], station=String[], phase=String[], time=DateTime[], amp=Float64[])
    else
        phases = DataFrame(arid=Int[], network=String[], station=String[], phase=String[], time=DateTime[])
    end
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    arid_idx = 1
    event_idx = Vector{Int}()
    offset = rand(rng, Uniform(Float64(15.), Float64(t_max-30.)), Int(n_events))

    mags = Float64.(rand(rng, Uniform(2.8e0, 3e0), n_events))
    amp_dist = Normal(-5.1e0, 1e0)

    phase_unc = Float64(params["phase_unc"])
    amp_unc = Float64(params["amp_unc"])
    L_t_unc = Normal(Float64(0.), phase_unc)
    L_amp_unc = Normal(Float64(0.), amp_unc)

    for i in 1:n_events
        origin_time = T0 + sec2date(offset[i])
        input = zeros(Float64, 7, 1)
        lat1 = Float64.(rand(rng, Uniform(params["lat_min"], params["lat_max"])))
        lon1 = Float64.(rand(rng, Uniform(params["lon_min"], params["lon_max"])))
        z1 = Float64.(rand(rng, Uniform(0.0, 50.0)))
        point_enu = trans(LLA(lat=Float64(lat1), lon=Float64(lon1)))
        input[1:3] = [point_enu.e, point_enu.n, z1*1e3]
        temp_phases = []
        temp_idx = []
        dists = []
        for row in eachrow(stations)
            input[4] = row.X
            input[5] = row.Y
            input[6] = row.Z
            R = sqrt((point_enu.e-row.X)^2 + (point_enu.n-row.Y)^2 + (z1-row.Z)^2)
            R = Float64.(R / 1000.0)
            if R > max_dist
                continue
            end
            if use_amp
                amp = GMPE(R, mags[i]) + rand(rng, L_amp_unc)
            end
            for phase_label in ["P", "S"]
                if phase_label == "P"
                    input[7] = 0e0
                else
                    input[7] = 1e0
                end
                input_normed = forward(input, scaler)
                T_pred = solve(input_normed, eikonet, scaler)[1] + rand(rng, L_t_unc)
                arrival_time = origin_time + sec2date(T_pred)
                if use_amp
                    push!(temp_phases, (arid_idx, row.network, row.station, phase_label, arrival_time, amp))
                else
                    push!(temp_phases, (arid_idx, row.network, row.station, phase_label, arrival_time))
                end
                push!(temp_idx, i)
                push!(dists, R)
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
        arrival_time = T0 + sec2date(rand(rng, Uniform(0e0, t_max)))
        push!(phases, (arid_idx, net, sta, phase_label, arrival_time, amp))
        push!(event_idx, 0)
        arid_idx += 1
    end
    return phases, event_idx
end

function evaluate_results(pfile)
    params = JSON.parsefile(pfile)

    stations = get_stations(params)
    if ~params["read_subset_only"]
        phases = CSV.read(params["phase_infile"], DataFrame)
    else
        phases = CSV.read(params["phase_infile"], DataFrame, limit=10000)
    end
    @printf("%d phases read in initially\n", nrow(phases))
    phases = innerjoin(phases, stations, on = [:network, :station])[:, names(phases)]
    @printf("%d phases remaining after removing those not in station list\n", nrow(phases))

    phases.phase = map(uppercase, phases.phase)
    sort!(phases, [:time])
    insertcols!(phases, :arid => 1:nrow(phases))
    println(first(phases, 10))

    assoc = CSV.read(@sprintf("%s_phase_iter_%d_phaseunc_%.02f_ampunc_%.02f_mdist_%s_edist_%s.csv",
                                params["label"], params["EM_epochs"], 
                                params["phase_unc"], params["amp_unc"], params["mstep_dist"], params["estep_dist"]), DataFrame)
    J_p, J_r = event_jaccard(assoc, phases)
    println("Neuma Jp: $J_p")
    println("Neuma Jr: $J_r")

    J_p, J_r = phase_jaccard(assoc, phases)
    println("Neuma Jp: $J_p")
    println("Neuma Jr: $J_r")

end