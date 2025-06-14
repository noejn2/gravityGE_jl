module GravityGE

using DataFrames, LinearAlgebra, Statistics

export gravityGE

function gravityGE(
    trade_data::DataFrame;
    theta::Real=4,
    beta_hat_name::Union{Nothing,String}=nothing,
    a_hat_name::Union{Nothing,String}=nothing,
    multiplicative::Bool=false
)

    tol = 1e-8
    max_iter = 1_000_000
    crit = 1.0

    required_cols = ["orig", "dest", "flow"]
    if !all(col -> col in names(trade_data), required_cols)
        error("Data set must contain columns 'orig', 'dest', and 'flow'.")
    end

    if nrow(trade_data) != length(unique(trade_data.orig .* trade_data.dest))
        error("Data set contains duplicate origin-destination pairs.")
    end

    if any(trade_data.flow .< 0)
        error("Negative flow values detected.")
    end

    N = sqrt(nrow(trade_data))
    if floor(N) != N
        error("Non-square data set detected. Size should be N×N.")
    end
    N = Int(N)

    sort!(trade_data, [:orig, :dest])
    trade_matrix = reshape(trade_data.flow, N, N)

    # Beta matrix
    if beta_hat_name !== nothing
        beta_vals = trade_data[!, beta_hat_name]
        beta_matrix = reshape(beta_vals, N, N)
        if any(diag(beta_matrix) .!= 0)
            error("Diagonal values of beta_hat must be zero.")
        end
        beta_matrix = exp.(beta_matrix)
        if any(beta_matrix .< 0)
            error("Negative beta values detected.")
        end
    else
        beta_matrix = ones(N, N)
    end

    # a_hat matrix
    if a_hat_name !== nothing
        a_vals = [unique(trade_data[trade_data.orig.==o, a_hat_name])[1] for o in unique(trade_data.orig)]
        if any(a_vals .< 0)
            error("Negative a_hat values detected.")
        end
        a_matrix = reshape(a_vals, N, 1)
    else
        a_matrix = ones(N, 1)
    end

    if minimum(diag(trade_matrix)) == 0
        @warn "Zero flow values detected on the diagonal."
    end
    replace!(trade_matrix, missing => 0.0)

    X = trade_matrix
    w_hat = ones(N)
    P_hat = ones(N)
    E = vec(sum(X, dims=1))  # Expenditure (column sum)
    Y = vec(sum(X, dims=2))  # Income (row sum)
    D = E - Y
    pi = X ./ transpose(repeat(E', N, 1))
    B = beta_matrix

    iter = 0
    while crit > tol && iter < max_iter
        iter += 1
        X_last_step = copy(X)

        w_hat = (a_matrix .* ((pi .* B) * (E ./ P_hat)) ./ Y) .^ (1 / (1 + theta))
        w_hat = w_hat .* (sum(Y) / sum(Y .* w_hat))

        P_hat = (pi' .* B') * (a_matrix .* (w_hat .^ (-theta)))

        E = multiplicative ? (Y .+ D) .* w_hat : Y .* w_hat .+ D

        pi_new = (pi .* B) .* transpose(repeat(a_matrix .* w_hat .^ (-theta), 1, N)) ./ transpose(repeat(P_hat, 1, N))
        X = pi_new .* transpose(repeat(E', N, 1))

        crit = maximum(abs.(log.(X) .- log.(X_last_step)))
    end

    if iter == max_iter
        @warn "Maximum iterations reached without convergence."
    end

    real_wage = w_hat ./ (P_hat .^ (-1 / theta))
    welfare = multiplicative ? real_wage : ((Y .* w_hat) .+ D) ./ (Y .+ D) ./ (P_hat .^ (-1 / theta))

    out1_df = DataFrame(
        orig=trade_data.orig,
        dest=trade_data.dest,
        new_trade=vec(X')
    )

    out2_df = DataFrame(
        orig=unique(trade_data.orig),
        welfare=welfare,
        nominal_wage=w_hat,
        price_index=P_hat .^ (-1 / theta)
    )

    return Dict("new_trade" => out1_df, "new_welfare" => out2_df)
end

end