function locate_events(params::Dict, X::Array{Float32}, M::Vector{Float32}, y_obs::Array{Float32}, prior_x::Distribution, eikonet::EikoNet1D)
    θ̂ = vec(mean(X[1:4,:], dims=2))
    push!(θ̂, mean(M))
    X_rec = X[5:end,:]
    θ̂_best = nothing
    σᵗ = Float32(params["phase_unc"])
    σᵃ = Float32(params["amp_unc"])

    function ℓπ(θ::AbstractArray)
        X_in = cat(repeat(θ[1:4], 1, size(X_rec, 2)), X_rec, dims=1)
        t_pred = dropdims(eikonet(X_in[2:end,:,:]), dims=1) .+ X_in[1,:]
        a_pred = GMPE.(eikonet.scale * hypo_dist(X_in), θ[5])

        ℓ_prior = logpdf(prior_x, X_in[2:4,1,1])
        if lowercase(params["mstep_dist"]) == "laplace"
            ℓL = sum(logpdf.(Laplace(0f0, σᵗ), t_pred-y_obs[:,1])) + sum(logpdf.(Laplace(0f0, σᵃ), a_pred-y_obs[:,2]))
        elseif lowercase(params["mstep_dist"]) == "huber"
            ℓL = sum(log_prob(HuberDensity(1.35f0), (t_pred-y_obs[:,1]) ./ σᵗ)) +
                 sum(log_prob(HuberDensity(1.35f0), (a_pred-y_obs[:,2]) ./ σᵃ))
        elseif lowercase(params["mstep_dist"]) == "normal"
            ℓL = sum(logpdf.(Normal(0f0, σᵗ), t_pred-y_obs[:,1])) + sum(logpdf.(Normal(0f0, σᵃ), a_pred-y_obs[:,2]))
        else
            error("Mdist not implemented")
        end
        return -ℓL - ℓ_prior
    end

    hz = BFGS(linesearch=LineSearches.HagerZhang())
    bt = BFGS(linesearch=LineSearches.BackTracking())
    options = Dict("normal" => Optim.Options(iterations=params["hypo_epochs"], g_tol=params["M_tol"]),
                   "huber" => Optim.Options(iterations=params["hypo_epochs"], g_tol=params["M_tol"]),
                   "laplace" => Optim.Options(iterations=params["hypo_epochs"], f_tol=params["M_tol"]))
    result = nothing
    try
        result = optimize(ℓπ, θ̂, hz, options[lowercase(params["mstep_dist"])], autodiff = :forward)
    catch
        result = optimize(ℓπ, θ̂, bt, options[lowercase(params["mstep_dist"])], autodiff = :forward)
    end
    θ̂ = Optim.minimizer(result)
    ℓ_best = Optim.minimum(result)
    return θ̂, ℓ_best
end

function locate_events(params::Dict, X::Array{Float32}, y_obs::Array{Float32}, γ_k::Array{Float32}, prior_x::Distribution, eikonet::EikoNet1D)
    θ̂ = vec(mean(X[1:4,:], dims=2))
    X_rec = X[5:end,:]
    θ̂_best = nothing

    function ℓπ(θ::AbstractArray)
        X_in = cat(repeat(θ[1:4], 1, size(X_rec, 2)), X_rec, dims=1)
        t_pred = dropdims(eikonet(X_in[2:end,:,:]), dims=1) .+ X_in[1,:]
        # X_cart = inverse(X_in[2:end,:,:], scaler)
        # ℓ_prior = -logpdf(prior_x, X_cart[2:4,1,1])
        if lowercase(params["mstep_dist"]) == "laplace"
            ℓ = Flux.mae(t_pred, y_obs[:,1], agg=identity)
        elseif lowercase(params["mstep_dist"]) == "huber"
            ℓ = Flux.huber_loss(t_pred ./ 0.5, y_obs[:,1] ./ 0.5; δ=1.35, agg=identity)
        elseif lowercase(params["mstep_dist"]) == "normal"
            ℓ = Flux.mse(t_pred, y_obs[:,1], agg=identity)
        else
            error("Mdist not implemented")
        end
        return sum(ℓ .* γ_k) / sum(γ_k)
    end

    hz = BFGS(linesearch=LineSearches.HagerZhang())
    bt = BFGS(linesearch=LineSearches.BackTracking())
    options = Dict("normal" => Optim.Options(iterations=params["hypo_epochs"], g_tol=params["M_tol"]),
                   "huber" => Optim.Options(iterations=params["hypo_epochs"], g_tol=params["M_tol"]),
                   "laplace" => Optim.Options(iterations=params["hypo_epochs"], f_tol=params["M_tol"]))
    result = nothing
    try
        result = optimize(ℓπ, θ̂, hz, options[lowercase(params["mstep_dist"])], autodiff = :forward)
    catch
        result = optimize(ℓπ, θ̂, bt, options[lowercase(params["mstep_dist"])], autodiff = :forward)
    end
    θ̂ = Optim.minimizer(result)
    ℓ_best = Optim.minimum(result)
    return θ̂, ℓ_best
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

function SBP_prior(N̂::AbstractArray, α::Float32)
    K = length(N̂)
    ν = zeros(Float32, K)
    φ = zeros(Float32, size(N̂))
    ν[K] = Float32(1.0)
    @inbounds for k in 1:(K-1)
        ν[k] = N̂[k] / (N̂[k] + α - Float32(1.0) + sum(N̂[(k+1):end]))
    end
    φ[1] = ν[1]
    @inbounds for k in 2:K
        φ[k] = ν[k]
        @inbounds for j in 1:(k-1)
            φ[k] *= (Float32(1.0) - ν[j])
        end
    end
    return φ
end

function SBP_prior(N̂::AbstractArray, idx::Vector{Int64}, α::Float32)
    K = length(N̂)
    ν = zeros(Float32, K)
    φ = zeros(Float32, size(N̂))
    ν[idx[K]] = Float32(1.0)
    @inbounds for k in 1:(K-1)
        ν[idx[k]] = N̂[idx[k]] / (N̂[idx[k]] + α - Float32(1.0) + sum(N̂[idx[(k+1):end]]))
    end
    φ[idx[1]] = ν[idx[1]]
    @inbounds for k in 2:K
        φ[idx[k]] = ν[idx[k]]
        @inbounds for j in 1:(k-1)
            φ[idx[k]] *= (Float32(1.0) - ν[idx[j]])
        end
    end
    return φ
end

function SBP_prior(N̂::AbstractArray, α::Float32, β::Float32) 
    K = length(N̂)
    ν = zeros(Float32, K)
    φ = zeros(Float32, size(N̂))
    ν[K] = Float32(1.0)
    @inbounds for k in 1:(K-1)
        ν[k] = (N̂[k] + α) / (α + β + sum(N̂[k:end]))
    end
    φ[1] = ν[1]
    @inbounds for k in 2:K
        φ[k] = ν[k]
        @inbounds for j in 1:(k-1)
            φ[k] *= (Float32(1.0) - ν[j])
        end
    end
    return φ
end

function m_step_hard!(params, X::Array{Float32}, M::Array{Float32}, y_obs::Array{Float32}, γ::Array{Float32},
                      z::Vector{Int}, prior_x::Distribution, eikonet::EikoNet)
    ℓL_tot = 0f0
    ϵ = 1f-8
    N̂ = Float32.(sum(γ, dims=1))
    N0 = Float32(params["N0"])
    if length(params["SBP_θ"]) == 0
        φ = (N̂ .+ N0) ./ sum(N̂ .+ N0)
    else
        idx = sortperm(vec(N̂), rev=true)
        φ = SBP_prior(N̂, idx, Float32.(params["SBP_θ"])...)
        φ = reshape(φ, 1, :)
    end
    K = size(X, 3)
    for k in unique(z)
        idx = findall(z .== k)
        θ̂, ℓL = locate_events(params, X[:,idx,k], M[:,k], y_obs[idx,:], prior_x, eikonet)
        X[1:4,:,k] .= θ̂[1:4]
        M[:,k] .= θ̂[5]
        ℓL_tot += ℓL
    end
    return N̂, φ
end


function m_step_soft!(params, X::Array{Float32}, M::Array{Float32}, y_obs::Array{Float32}, γ::Array{Float32}, prior_x::Distribution, eikonet::EikoNet)
    ℓL_tot = 0f0
    ϵ = 1f-8
    γ .+= ϵ
    N̂ = Float32.(sum(γ, dims=1))
    if length(params["SBP_θ"]) == 0
        φ = N̂ ./ sum(N̂)
    else
        φ = SBP_prior(N̂, Float32.(params["SBP_θ"])...)
    end
    K = size(X, 3)
    σᵗ = fill(Float32(params["phase_unc"]), K)
    σᵃ = fill(Float32(params["amp_unc"]), K)
    for k in 1:K
        θ̂, ℓL = locate_events(params, X[:,:,k], y_obs[:,:], γ[:,k], prior_x, eikonet)
        X[1:4,:,k] .= θ̂[1:4]
        M[:,k] .= sum(compute_mag.(eikonet.scale * hypo_dist(X[:,:,k]), y_obs[:,2]) .* γ[:,k]) / N̂[k]
        ℓL_tot += ℓL
    end
    return N̂, φ, σᵗ, σᵃ
end

function e_step_soft(params, X::Array{Float32}, M::Array{Float32}, y_obs::Array{Float32}, φ::Array{Float32}, eikonet::EikoNet, σᵗ::Vector{Float32}, σᵃ::Vector{Float32})
    ϵ = 1f-8
    N = size(y_obs, 1)
    K = length(φ)
    tt_pred = dropdims(eikonet(X[2:end,:,:]), dims=1)
    t_pred = tt_pred + X[1,:,:]
    a_pred = GMPE.(eikonet.scale * hypo_dist(X), M)
    ℓπ = zeros(Float32, N, K)
    δt = t_pred .- y_obs[:,1]
    δa = a_pred .- y_obs[:,2]
    for k in 1:K
        if lowercase(params["estep_dist"]) == "laplace"
            ℓπ[:,k] .= logpdf.(Laplace(0f0, σᵗ[k]), δt[:,k]) .+ logpdf.(Laplace(0f0, σᵃ[k]), δa[:,k])
        elseif lowercase(params["estep_dist"]) == "normal"
            ℓπ[:,k] .= logpdf.(Normal(0f0, σᵗ[k]), δt[:,k]) .+ logpdf.(Normal(0f0, σᵃ[k]), δa[:,k])
        else
            error("Estep distribution not defined")
        end
    end
    γ = φ .* exp.(ℓπ)
    γ[isnan.(γ)] .= 0f0
    γ .+= ϵ
    ℓ_new = -sum(log.(sum(γ, dims=2)))
    γ ./= sum(γ, dims=2)
    return γ, ℓ_new
end

function e_step_hard(params, X::Array{Float32}, M::Array{Float32}, y_obs::Array{Float32}, φ::Array{Float32}, prior_x::Distribution,
                     eikonet::EikoNet, σᵗ::Vector{Float32}, σᵃ::Vector{Float32})
    ϵ = 1f-8
    N = size(y_obs, 1)
    K = length(φ)
    tt_pred = dropdims(eikonet(X[2:end,:,:]), dims=1)
    t_pred = tt_pred + X[1,:,:]
    a_pred = GMPE.(eikonet.scale * hypo_dist(X), M)
    ℓπ = zeros(Float32, N, K)
    δt = t_pred .- y_obs[:,1]
    δa = a_pred .- y_obs[:,2]
    for k in 1:K
        if lowercase(params["estep_dist"]) == "laplace"
            ℓπ[:,k] .= logpdf.(Laplace(0f0, σᵗ[k]), δt[:,k]) .+ logpdf.(Laplace(0f0, σᵃ[k]), δa[:,k])
        elseif lowercase(params["estep_dist"]) == "normal"
            ℓπ[:,k] .= logpdf.(Normal(0f0, σᵗ[k]), δt[:,k]) .+ logpdf.(Normal(0f0, σᵃ[k]), δa[:,k])
        elseif lowercase(params["estep_dist"]) == "huber"
            ℓπ[:,k] .= log_prob(HuberDensity(1.35f0), δt[:,k] ./ σᵗ[k]) .+ log_prob(HuberDensity(1.35f0), δa[:,k] ./ σᵃ[k])
        else
            error("Estep distribution not defined")
        end
    end
    ℓ_prior = logpdf(prior_x, X[2:4,1,:])

    log_γ = log.(φ .+ ϵ) .+ ℓπ  .+ transpose(ℓ_prior)
    log_γ[isnan.(log_γ)] .= -9f10
    γ = zeros(Float32, size(M))
    idx = argmax(log_γ, dims=2)
    γ[idx] .= 1f0
    z = vec([x[2] for x in idx])
    ℓ_new = -sum(log_γ[i,z[i]] for i in 1:N)
    return γ, z, ℓ_new
end

function setup_priors(params, scaler::MinmaxScaler)
    prior_μ = zeros(Float32, 3)
    prior_μ[1:2] .= 5f-1
    prior_μ[3] = Float32((params["prior_depth_mean"] - scaler.min[3]) / scaler.scale)

    prior_Σ = Float32.(params["prior_cov"] / scaler.scale)
    prior_x = MvNormal(prior_μ, prior_Σ)
    return prior_x
end

function associate(params, X::Array{Float32}, y_obs::Array{Float32}, eikonet::EikoNet)

    N = size(y_obs, 1)
    K = size(X, 3)
    y_obs = y_obs[:,:,1]

    idx_best = 0
    ε = params["E_tol"]

    ℓ_new = Inf
    ℓ_best = Inf

    scaler = data_scaler(params)
    X[2:end,:,:] = forward(X[2:end,:,:], scaler)

    prior_x = setup_priors(params, scaler)

    φ = Flux.unsqueeze(Float32.(ones(Float32, K) ./ K), 1)
    γ = fill(Float32(1.0/K), N, K)
    z = vec([x[2] for x in argmax(γ, dims=2)])
    N̂ = sum(γ, dims=1)
    M = zeros(Float32, N, K)
    X_best = M_best = γ_best = nothing

    losses = zeros(Float32, params["EM_epochs"])
    σᵗ = fill(Float32(params["phase_unc"]), K)
    σᵃ = fill(Float32(params["amp_unc"]), K)
    for epoch in 1:params["EM_epochs"]

        z_old = z
        γ, z, ℓ_new = e_step_hard(params, X, M, y_obs, φ, prior_x, eikonet, σᵗ, σᵃ)
        N̂, φ = m_step_hard!(params, X, M, y_obs, γ, z, prior_x, eikonet)
        losses[epoch] = ℓ_new

        if params["verbose"]
            println("epoch $epoch -ℓπ $ℓ_new ", Int.(round.(N̂)))
        end
        
        if z == z_old
            break
        end
    end

    t_pred = dropdims(eikonet(X[2:end,:,:]), dims=1) + X[1,:,:]
    a_pred = GMPE.(eikonet.scale * hypo_dist(X), M)
    y_pred = cat(t_pred, a_pred, dims = 3)
    y_pred = permutedims(y_pred, (1, 3, 2))
    resid = y_obs .- y_pred

    z = vec([x[2] for x in argmax(γ, dims=2)])
    γ_best_cond = zeros(Float32, N, K)

    X[2:end,:,:] = inverse(X[2:end,:,:], scaler)
    X = mean(X, dims=2)[1:4,1,:]

    X[2:4,:] .*= 1f3

    return X, M, z, γ_best_cond, resid
end