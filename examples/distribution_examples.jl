# Velocity Distribution Functions - Visualization Examples
# This script demonstrates sampling from distributions and comparing with theoretical PDFs

using VelocityDistributionFunctions
using CairoMakie
using LinearAlgebra: norm, dot
using Statistics: mean, var

# Set figure theme
set_theme!(Theme(fontsize=14, size=(800, 600)))

## Example 1: Maxwellian Speed Distribution
println("Generating Maxwellian speed distribution example...")

# Create Maxwellian distribution
d_max = Maxwellian(1.0)

# Generate samples
n_samples = 100000
samples_max = rand(d_max, n_samples)
speeds_max = [norm(samples_max[:, i]) for i in 1:n_samples]

# Theoretical speed distribution: f(v) = 4π v² f₃D(v)
v_range = range(0, 4, length=200)
speed_pdf_max = [4π * v^2 * pdf(d_max, [v, 0, 0]) for v in v_range]

# Plot
fig1 = Figure()
ax1 = Axis(fig1[1, 1],
    xlabel="Speed (vth units)",
    ylabel="Probability Density",
    title="Maxwellian Speed Distribution")

hist!(ax1, speeds_max, bins=100, normalization=:pdf,
      label="Sampled", color=(:blue, 0.5))
lines!(ax1, v_range, speed_pdf_max,
       color=:red, linewidth=3, label="Theoretical")
axislegend(ax1, position=:rt)

save("maxwellian_speeds.png", fig1)
display(fig1)

## Example 2: Comparing Maxwellian and Kappa Distributions
println("Generating Maxwellian vs Kappa comparison...")

# Create distributions
d_kappa = Kappa(3.0, 1.0)

# Generate samples
samples_kappa = rand(d_kappa, n_samples)
speeds_kappa = [norm(samples_kappa[:, i]) for i in 1:n_samples]

# Theoretical distributions
v_range_extended = range(0, 6, length=200)
speed_pdf_max_ext = [4π * v^2 * pdf(d_max, [v, 0, 0]) for v in v_range_extended]
speed_pdf_kappa = [4π * v^2 * pdf(d_kappa, [v, 0, 0]) for v in v_range_extended]

# Plot on log scale to show tails
fig2 = Figure()
ax2 = Axis(fig2[1, 1],
    xlabel="Speed (vth units)",
    ylabel="Probability Density",
    title="Maxwellian vs Kappa (κ=3) - Heavy Tails",
    yscale=log10)

hist!(ax2, speeds_max, bins=100, normalization=:pdf,
      label="Maxwellian samples", color=(:blue, 0.4))
hist!(ax2, speeds_kappa, bins=100, normalization=:pdf,
      label="Kappa samples", color=(:red, 0.4))
lines!(ax2, v_range_extended, speed_pdf_max_ext,
       color=:blue, linewidth=2.5, label="Maxwellian theory")
lines!(ax2, v_range_extended, speed_pdf_kappa,
       color=:red, linewidth=2.5, label="Kappa theory")

axislegend(ax2, position=:rt)
ylims!(ax2, 1e-5, 2)

save("maxwellian_vs_kappa.png", fig2)
display(fig2)

## Example 3: BiMaxwellian Anisotropy
println("Generating BiMaxwellian anisotropy example...")

# Create anisotropic BiMaxwellian (cold perp, hot parallel)
d_bimaxwellian = BiMaxwellian(0.5, 2.0)

# Generate samples
samples_bi = rand(d_bimaxwellian, n_samples)

# Decompose into parallel and perpendicular components
B_dir = [0, 0, 1]
v_par = [dot(samples_bi[:, i], B_dir) for i in 1:n_samples]
v_perp = [norm(samples_bi[:, i] - dot(samples_bi[:, i], B_dir) * B_dir)
          for i in 1:n_samples]

# Plot 2D distribution
fig3 = Figure()
ax3 = Axis(fig3[1, 1],
    xlabel="v∥ (parallel to B)",
    ylabel="v⊥ (perpendicular to B)",
    title="BiMaxwellian: vth⊥=0.5, vth∥=2.0",
    aspect=1)

hexbin!(ax3, v_par, v_perp, bins=50, colormap=:viridis)
Colorbar(fig3[1, 2], label="Counts")

# Add reference circles
scatter!(ax3, [0], [0], color=:white, markersize=5, label="Origin")

save("bimaxwellian_anisotropy.png", fig3)
display(fig3)

## Example 4: Kappa Parameter Sweep
println("Generating Kappa parameter sweep...")

kappa_values = [2.5, 5.0, 10.0, 50.0]
colors = [:red, :orange, :green, :blue]

fig4 = Figure()
ax4 = Axis(fig4[1, 1],
    xlabel="Speed (vth units)",
    ylabel="Probability Density",
    title="Effect of κ on Tail Heaviness",
    yscale=log10)

v_range_tails = range(0, 8, length=200)

for (κ, color) in zip(kappa_values, colors)
    d_k = Kappa(κ, 1.0)

    # Generate samples
    samples_k = rand(d_k, 50000)
    speeds_k = [norm(samples_k[:, i]) for i in 1:50000]

    # Theoretical PDF
    speed_pdf_k = [4π * v^2 * pdf(d_k, [v, 0, 0]) for v in v_range_tails]

    # Plot
    hist!(ax4, speeds_k, bins=100, normalization=:pdf,
          color=(color, 0.3))
    lines!(ax4, v_range_tails, speed_pdf_k,
           color=color, linewidth=2.5, label="κ=$κ")
end

# Add Maxwellian reference
speed_pdf_ref = [4π * v^2 * pdf(d_max, [v, 0, 0]) for v in v_range_tails]
lines!(ax4, v_range_tails, speed_pdf_ref,
       color=:black, linewidth=2, linestyle=:dash, label="Maxwellian (κ→∞)")

axislegend(ax4, position=:rt)
ylims!(ax4, 1e-7, 2)

save("kappa_parameter_sweep.png", fig4)
display(fig4)

## Example 5: 1D Velocity Component Distributions
println("Generating 1D velocity component comparison...")

# Compare vx distributions for all three types
fig5 = Figure()
ax5 = Axis(fig5[1, 1],
    xlabel="vx (one velocity component)",
    ylabel="Probability Density",
    title="1D Velocity Component Distributions")

# Extract x-components
vx_max = samples_max[1, :]
vx_kappa = samples_kappa[1, :]
vx_bi = samples_bi[1, :]

# Theoretical 1D distributions
vx_range = range(-4, 4, length=200)

# For Maxwellian: 1D projection is Gaussian
pdf_1d_max = [exp(-vx^2 / (2 * d_max.vth^2)) / sqrt(2π * d_max.vth^2) for vx in vx_range]

# For BiMaxwellian (x is perpendicular to B=[0,0,1])
pdf_1d_bi = [exp(-vx^2 / (2 * d_bimaxwellian.vth_perp^2)) / sqrt(2π * d_bimaxwellian.vth_perp^2)
             for vx in vx_range]

# Plot
hist!(ax5, vx_max, bins=100, normalization=:pdf,
      label="Maxwellian", color=(:blue, 0.4))
hist!(ax5, vx_kappa, bins=100, normalization=:pdf,
      label="Kappa (κ=3)", color=(:red, 0.4))
hist!(ax5, vx_bi, bins=100, normalization=:pdf,
      label="BiMaxwellian", color=(:green, 0.4))

lines!(ax5, vx_range, pdf_1d_max, color=:blue, linewidth=2)
lines!(ax5, vx_range, pdf_1d_bi, color=:green, linewidth=2, linestyle=:dash)

axislegend(ax5, position=:rt)

save("velocity_component_comparison.png", fig5)
display(fig5)

println("\nAll examples generated successfully!")
println("Saved figures:")
println("  - maxwellian_speeds.png")
println("  - maxwellian_vs_kappa.png")
println("  - bimaxwellian_anisotropy.png")
println("  - kappa_parameter_sweep.png")
println("  - velocity_component_comparison.png")

# Print some statistics
println("\nSample statistics:")
println("Maxwellian:")
println("  Mean speed: $(round(mean(speeds_max), digits=3)) (theoretical: $(round(2/sqrt(π), digits=3)))")
println("  Variance: $(round(var(vx_max), digits=3)) (theoretical: 1.0)")

println("\nKappa (κ=3):")
println("  Mean speed: $(round(mean(speeds_kappa), digits=3))")
println("  Variance: $(round(var(vx_kappa), digits=3)) (has heavier tails)")

println("\nBiMaxwellian:")
println("  ⟨v∥⟩ = $(round(mean(v_par), digits=3)), var(v∥) = $(round(var(v_par), digits=3)) (expected: $(d_bimaxwellian.vth_parallel^2))")
println("  ⟨v⊥²⟩ = $(round(mean(v_perp.^2), digits=3)) (expected: $(2*d_bimaxwellian.vth_perp^2))")
