using Dates
using JSON

function convert_gamma_to_hyposvi(fname, outfile)
    picks = CSV.read(fname, DataFrame)
    nets = []
    stas = []
    chans = []
    evids = []
    for row in eachrow(picks)
        push!(evids, row.event_index)
        net, sta, loc, chan = split(row.station_id, ".")
        push!(nets, net)
        push!(stas, sta)
    end
    phases = DataFrame(evid=evids, network=nets, station=stas, phase=picks[:,:phase_type], time=picks[:,:phase_time])
    unique!(phases)
    filter!(row -> row.evid >= 0.0, phases)
    unique!(phases, [:evid, :network, :station, :phase])
    CSV.write(outfile, phases)
end

function read_syn_dataset(params)
    sta_dict = JSON.parsefile("synthetic_stations.json")
    stations = DataFrame(network=String[], station=String[], latitude=Float32[], longitude=Float32[], elevation=Float32[])
    for key in keys(sta_dict)
        net, sta, loc, chan = split(key, ".")
        elev = sta_dict[key]["elevation(m)"]
        lat = sta_dict[key]["latitude"]/1000.
        lon = sta_dict[key]["longitude"]
        push!(stations, (net, sta, lat, lon, elev))
    end
    events = CSV.read("synthetic_events.csv", DataFrame)
    picks = CSV.read("synthetic_picks.csv", DataFrame)
    nets = []
    stas = []
    chans = []
    for row in eachrow(picks)
        net, sta, loc, chan = split(row.station_id, ".")
        push!(nets, net)
        push!(stas, sta)
    end
    phases = DataFrame(network=nets, station=stas, phase=picks[:,:phase_type], time=picks[:,:phase_time])
    
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    X = Vector{Float32}()
    Y = Vector{Float32}()
    Z = Vector{Float32}()
    for row in eachrow(stations)
        xyz = trans(LLA(lat=row.latitude, lon=row.longitude))
        push!(X, xyz.e)
        push!(Y, xyz.n)
        push!(Z, row.elevation*1000.0)
    end
    stations= hcat(stations, DataFrame(X=X, Y=Y, Z=Z))


    return phases, stations, events
end

function sec2date(s::AbstractFloat)
    sec_sign = Int32(sign(s))
    s = abs(s)
    sec = Int32(floor(s))
    msec = Int32(floor(1000*(s - sec)))
    return Dates.Second(sec * sec_sign) + Dates.Millisecond(msec * sec_sign)
end

struct Origin
    lat::Float32
    lon::Float32
    depth::Float32
    time::Dates.DateTime
    mag::Float32
    unc_z::Float32
    X::Float32
    Y::Float32
    arids::Vector
end

function build_eikonet_syn_params()
    params = Dict()
    params["station_file"] = "/scratch/zross/oak_ridge/scsn_stations.csv"
    params["velmod_file"] = "/home/zross/git/eikonet_julia/eikonet_hk77/hk77.csv"
    params["lon_min"] = -118.50
    params["lon_max"] = -116.50
    params["lat_min"] = 34.8
    params["lat_max"] = 36.8
    params["z_min"] = -5.0
    params["z_max"] = 50.0
    params["model_file"] = "/home/zross/git/eikonet_julia/eikonet_hk77/model.bson"
    params["n_epochs"] = 200
    params["lr"] = 1e-3
    return params
end

function build_hyposvi_params()
    params = Dict()
    params["phase_file"] = "/scratch/zross/oak_ridge/scsn_oak_ridge.csv"
    params["station_file"] = "/scratch/zross/oak_ridge/scsn_stations.csv"
    params["velmod_file"] = "/scratch/zross/oak_ridge/vz_socal.csv"
    params["catalog_outfile"] = "/scratch/zross/oak_ridge/catalog_svi.csv"
    params["lon_min"] = -119.8640
    params["lon_max"] = -117.8640
    params["lat_min"] = 33.3580
    params["lat_max"] = 35.3580
    params["z_min"] = 0.0
    params["z_max"] = 60.0
    params["model_file"] = "/scratch/zross/oak_ridge/model.bson"
    params["n_epochs"] = 1000
    params["n_particles"] = 1
    params["lr"] = 1e-3
    params["phase_unc"] = 0.20
    params["verbose"] = true
    params["k-NN"] = 500
    params["iter_tol"] = 1e-2
    params["max_k-NN_dist"] = 50
    params["n_ssst_iter"] = 3
    params["inversion_method"] = "SVI"
    return params
end

function build_neuma_params()
    params = Dict()
    params["phase_file"] = "/scratch/zross/oak_ridge/scsn_oak_ridge.csv"
    params["station_file"] = "/home/zross/git/eikonet_julia/scsn_stations.csv"
    params["velmod_file"] = "/scratch/zross/oak_ridge/vz_socal.csv"
    params["catalog_outfile"] = "/scratch/zross/oak_ridge/catalog_svi.csv"
    params["lon_min"] = -119.8640
    params["lon_max"] = -117.8640
    params["lat_min"] = 33.3580
    params["lat_max"] = 35.3580
    params["z_min"] = 0.0
    params["z_max"] = 60.0
    params["model_file"] = "/scratch/zross/oak_ridge/model.bson"
    params["n_epochs"] = 500
    params["n_particles"] = 10
    params["lr"] = 1e-3
    params["phase_unc"] = 0.5
    params["verbose"] = true
    params["k-NN"] = 500
    params["iter_tol"] = 1e-2
    params["max_k-NN_dist"] = 50
    params["n_ssst_iter"] = 1
    params["n_det"] = 4
    params["inversion_method"] = "EM"
    return params
end

function build_neuma_syn_params()
    params = Dict()
    params["phase_file"] = "/scratch/zross/oak_ridge/scsn_oak_ridge.csv"
    params["station_file"] = "/home/zross/git/eikonet_julia/scsn_stations.csv"
    params["velmod_file"] = "/scratch/zross/oak_ridge/vz_socal.csv"
    params["catalog_outfile"] = "/scratch/zross/oak_ridge/catalog_svi.csv"
    params["lon_min"] = -118.004
    params["lon_max"] = -117.004
    params["lat_min"] = 35.205
    params["lat_max"] = 36.205
    params["z_min"] = 0.0
    params["z_max"] = 60.0
    params["model_file"] = "/home/zross/git/eikonet_julia/syntest/model.bson"
    params["n_epochs"] = 200
    params["n_clusters"] = 10
    params["lr"] = 1e-3
    params["phase_unc"] = 0.20
    params["amp_unc"] = 1.0
    params["verbose"] = true
    params["k-NN"] = 500
    params["iter_tol"] = 1e-5
    params["n_det"] = 15
    params["n_warmup_iter"] = 20
    params["n_iter"] = 10
    params["huber_delta"] = 0.5
    return params
end

function build_neuma_data_params()
    params = Dict()
    params["catalog_outfile"] = "/scratch/zross/oak_ridge/catalog_svi.csv"
    params["station_file"] = "/home/zross/git/eikonet_julia/scsn_stations.csv"
    params["velmod_file"] = "/home/zross/git/eikonet_julia/eikonet_hk77/hk77.csv"
    params["lon_min"] = -118.50
    params["lon_max"] = -116.50
    params["lat_min"] = 34.8
    params["lat_max"] = 36.8
    params["z_min"] = -5.0
    params["z_max"] = 50.0
    params["model_file"] = "/home/zross/git/eikonet_julia/eikonet_hk77/model.bson"
    params["n_epochs"] = 300
    params["n_warmup_iter"] = 30
    params["n_iter"] = 30
    params["lr"] = 1e-3
    params["phase_unc"] = 0.35
    params["amp_unc"] = 1.0
    params["verbose"] = true
    params["iter_tol"] = 1e-5
    params["n_det"] = 10
    params["huber_delta"] = 0.5
    return params
end

function fwd_transform(point::Geodesy.LLA, origin::Geodesy.LLA)
    enu = ENU(point, origin, wgs84) ./ 1f3
    return enu
end

function fwd_transform!(X::Array, origin::Geodesy.LLA)
    for i in 1:size(X, 2)
        x_tmp = fwd_transform(LLA(lon=X[1], lat=X[1]), origin)
        X[1,i] = x_tmp.e
        X[2,i] = x_tmp.n
        x_tmp = fwd_transform(LLA(lon=X[4], lat=X[4]), origin)
        X[4,i] =  x_tmp.e
        X[5,i] = x_tmp.n
    end
end

struct MinmaxScaler
    min::Float32
    scale::Float32
end

function fit(X::AbstractArray, ::Type{MinmaxScaler})
    mins = minimum(X[1:6,:])
    maxs = maximum(X[1:6,:])
    return MinmaxScaler(mins, maxs-mins)
end

function forward!(X::AbstractArray, scaler::MinmaxScaler)
    X[1:6,:] .= (X[1:6,:] .- scaler.min) ./ scaler.scale
end

function forward(X::AbstractArray, scaler::MinmaxScaler)
    X_new = (X[1:6,:] .- scaler.min) ./ scaler.scale
    return cat(X_new, reshape(X[7,:], 1, :), dims=1)
end

function forward_point(X::AbstractArray, scaler::MinmaxScaler)
    return (X .- scaler.min) ./ scaler.scale
end

function inverse_point(Y::AbstractArray, scaler::MinmaxScaler)
    return (Y .* scaler.scale) .+ scaler.min
end

function inverse!(Y::AbstractArray, scaler::MinmaxScaler)
    Y[1:6,:] .= (Y[1:6,:] .* scaler.scale) .+ scaler.min
end

function inverse(Y::AbstractArray, scaler::MinmaxScaler)
    Y_new = (Y[1:6,:] .* scaler.scale) .+ scaler.min
    return cat(Y_new, reshape(Y[7,:], 1, :), dims=1)
end

function get_stations(params)
    # Note: origin_lla must match what the Eikonet was trained with.
    stations = CSV.read(params["station_file"], DataFrame)
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    X = Vector{Float32}()
    Y = Vector{Float32}()
    Z = Vector{Float32}()
    idx = []
    count = 0
    for row in eachrow(stations)
        count += 1
        xyz = trans(LLA(lat=row.latitude, lon=row.longitude))
        if (row.latitude < params["lat_min"]) || (row.latitude > params["lat_max"]) ||
           (row.longitude < params["lon_min"]) || (row.longitude > params["lon_max"])
            continue
        end
        push!(X, xyz.e)
        push!(Y, xyz.n)
        push!(Z, row.elevation*1000.0)
        push!(idx, count)
    end
    stations = stations[idx,:]
    return unique(hcat(stations, DataFrame(X=X, Y=Y, Z=Z)), [:network, :station])
end

function data_scaler(params)
    x = zeros(Float32, 7, 2)
    min_lla = LLA(lat=params["lat_min"], lon=params["lon_min"])
    max_lla = LLA(lat=params["lat_max"], lon=params["lon_max"])
    origin = LLA(lat=params["lat_min"], lon=params["lon_min"])
    trans = ENUfromLLA(origin, wgs84)
    min_enu = trans(min_lla)
    max_enu = trans(max_lla)
    x[1,1] = min_enu.e
    x[2,1] = min_enu.n
    x[3,1] = params["z_min"] * 1f3
    x[4,1] = min_enu.e
    x[5,1] = min_enu.n
    x[6,1] = params["z_min"] * 1f3
    x[7,1] = 0f0

    x[1,2] = max_enu.e
    x[2,2] = max_enu.n
    x[3,2] = params["z_max"] * 1f3
    x[4,2] = max_enu.e
    x[5,2] = max_enu.n
    x[6,2] = params["z_max"] * 1f3
    x[7,2] = 1f0

    scaler = fit(x, MinmaxScaler)
    return scaler
end

function timedelta(t1::DateTime, t2::DateTime)
    # returns total seconds between t1,t2
    (t1-t2) / Millisecond(1000)
end

function format_arrivals(phases::DataFrame, stations::DataFrame)
    phase_sta = innerjoin(phases, stations, on = [:network, :station])
    X_inp = zeros(Float32, size(phase_sta, 1), 4)
    X_inp[:,1] .= phase_sta.X
    X_inp[:,2] .= phase_sta.Y
    X_inp[:,3] .= phase_sta.Z
    arrival_times = DateTime.(phase_sta[!, "time"])
    T_obs = zeros(Float32, 1, length(arrival_times))
    amp_included = false
    if "amp" in names(phase_sta)
        amp_included = true
    end
    amp_obs = zeros(Float32, size(T_obs))
    for (i, row) in enumerate(eachrow(phase_sta))
        if row.phase == "P"
            X_inp[i,4] = 0
        elseif row.phase == "S"
            X_inp[i,4] = 1
        else
            println("Error: unknown Phase label (not P or S). Exiting...")
            println(row)
        end
        T_obs[i] = timedelta(arrival_times[i], minimum(arrival_times))
        if amp_included
            amp_obs[i] = phase_sta.amp[i]
        end
    end
    T_ref = minimum(arrival_times)
    if amp_included
        return X_inp, T_obs, amp_obs, T_ref, phase_sta
    else
        return X_inp, T_obs, T_ref, phase_sta
    end
end
