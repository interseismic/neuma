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
    params["catalog_outfile"] = "/scratch/zross/eikonet_julia/catalog_neuma.csv"
    params["station_file"] = "/home/zross/git/eikonet_julia/scsn_stations.csv"
    params["velmod_file"] = "/home/zross/git/eikonet_julia/eikonet_hk77/hk77.csv"
    params["lon_min"] = -118.50
    params["lon_max"] = -116.50
    params["lat_min"] = 34.8
    params["lat_max"] = 36.8
    params["z_min"] = -5.0
    params["z_max"] = 50.0
    params["model_file"] = "/home/zross/git/eikonet_julia/eikonet_hk77/model.bson"
    params["n_epochs"] = 100
    params["n_warmup_iter"] = 30
    params["n_iter"] = 10
    params["lr"] = 1e-3
    params["phase_unc"] = 0.50
    params["amp_unc"] = 1.0
    params["verbose"] = false
    params["iter_tol"] = 1e-5
    params["n_det"] = 5
    params["huber_delta"] = 0.10
    return params
end