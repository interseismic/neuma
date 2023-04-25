function build_dataset_only(pfile, outcat_fname; seed=1, n_events=1)
    params = JSON.parsefile(pfile)

    model = BSON.load(params["model_file"], @__MODULE__)[:model]
    scaler = data_scaler(params)

    stations = get_stations(params)
    phases, event_idx = generate_syn_dataset(params, stations, model, scaler;
        n_events=n_events, n_fake=0, seed=seed, max_picks_per_event=Inf, t_max=86400.0, use_amp=false)
    insertcols!(phases, :evid => event_idx)
    CSV.write(outcat_fname, phases)
end