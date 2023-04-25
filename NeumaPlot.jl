function plot_events(origins::DataFrame)
    scatter(origins[!,:lon], origins[!,:lat], left_margin = 20Plots.mm)
    origins = CSV.read("/scratch/zross/oak_ridge/scsn_cat.csv", DataFrame)
    scatter!(origins[!,:lon], origins[!,:lat], left_margin = 20Plots.mm)
    savefig("events.png")
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
