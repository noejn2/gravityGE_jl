using Test
using DataFrames
using GravityGE  # This should now correctly load gravityGE

@testset "Checks welfare respond according to theoretical expectations" begin
    orig = repeat(string.('A':'Z'), 26)
    dest = repeat(string.('A':'Z'), inner=26)
    flows = DataFrame(orig=orig, dest=dest, flow=ones(26 * 26))

    # No change
    out = gravityGE(flows)
    @test out["new_welfare"].welfare ≈ ones(26)

    # More bitrade costs
    flows.bitrade_costs = fill(-4 * log(1.5), 26 * 26)
    for i in 1:26
        flows.bitrade_costs[(i-1)*26+i] = 0
    end
    out = gravityGE(flows, beta_hat_name="bitrade_costs")
    @test all(out["new_welfare"].welfare .<= 1.0)

    # Less bitrade costs
    flows.bitrade_costs .= -4 * log(0.5)
    for i in 1:26
        flows.bitrade_costs[(i-1)*26+i] = 0
    end
    out = gravityGE(flows, beta_hat_name="bitrade_costs")
    @test all(out["new_welfare"].welfare .>= 1.0)

    # More productivity
    flows.prod = fill(2.0, 26 * 26)
    out = gravityGE(flows, a_hat_name="prod")
    @test all(out["new_welfare"].welfare .>= 1.0)

    # Less productivity
    flows.prod .= 0.5
    out = gravityGE(flows, a_hat_name="prod")
    @test all(out["new_welfare"].welfare .<= 1.0)

    # Multiplicative: No change
    out = gravityGE(flows, multiplicative=true)
    @test out["new_welfare"].welfare ≈ ones(26)

    # Multiplicative: More bitrade costs
    flows.bitrade_costs .= -4 * log(1.5)
    for i in 1:26
        flows.bitrade_costs[(i-1)*26+i] = 0
    end
    out = gravityGE(flows, beta_hat_name="bitrade_costs", multiplicative=true)
    @test all(out["new_welfare"].welfare .<= 1.0)

    # Multiplicative: Less bitrade costs
    flows.bitrade_costs .= -4 * log(0.5)
    for i in 1:26
        flows.bitrade_costs[(i-1)*26+i] = 0
    end
    out = gravityGE(flows, beta_hat_name="bitrade_costs", multiplicative=true)
    @test all(out["new_welfare"].welfare .>= 1.0)

    # Multiplicative: More productivity
    flows.prod .= 2.0
    out = gravityGE(flows, a_hat_name="prod", multiplicative=true)
    @test all(out["new_welfare"].welfare .>= 1.0)

    # Multiplicative: Less productivity
    flows.prod .= 0.5
    out = gravityGE(flows, a_hat_name="prod", multiplicative=true)
    @test all(out["new_welfare"].welfare .<= 1.0)
end
