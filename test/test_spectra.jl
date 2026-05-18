@testset "sum_valid! invariants" begin
    # Synthetic NaN handling: invalid entries must be excluded from both num & den.
    Sʹ = Float32[1 2; 3 4;;; 5 NaN; 7 8]  # 2×2×2
    dΩ = Float32[1.0 0.5; 0.5 1.0]        # 2×2
    Ω = zeros(Float32, 2, 2)
    flux = VelocityDistributionFunctions.sum_valid!(Sʹ, dΩ, Ω)
    # For (E=1, t=1): num = 1*1 + 3*0.5 = 2.5, den = 1+0.5 = 1.5 ⇒ 5/3
    @test flux[1, 1] ≈ 2.5 / 1.5
    # For (E=2, t=1): num = 2*1 + 4*0.5 = 4, den = 1+0.5 = 1.5
    @test flux[2, 1] ≈ 4 / 1.5
    # For (E=2, t=2): S[1, 2, 2] = NaN ⇒ excluded; num = 8 * 1.0 = 8, den = 1.0
    @test flux[2, 2] ≈ 8 / 1.0
    @test Ω[2, 2] ≈ 1.0  # only one valid sample contributed
end
