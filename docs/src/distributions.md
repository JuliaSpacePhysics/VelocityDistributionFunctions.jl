# Velocity Distribution Functions

This package provides implementations of common velocity distribution functions used in plasma physics.

```@docs; canonical=false
Maxwellian
BiMaxwellian
Kappa
```

For more details about Kappa distribution, see [Kappa](kappa.md).

## Maxwellian Distribution

The isotropic Maxwellian distribution is the most basic equilibrium distribution:

```@example vdf
using VelocityDistributionFunctions

v_th = 1.0 # Thermal velocity
u0 = [0.5, 0.0, 0.0] # Drift velocity
vdf = Maxwellian(v_th, u0)

# Sample from the distribution
samples = rand(vdf, 10000)

# Evaluate Theoretical PDF
𝐯 = [1.0, 0.0, 0.0]
vdf(𝐯)
```

Unit is also supported

```@example vdf
using Unitful

T = 30000u"K" # Temperature
vdf = Maxwellian(T)

𝐯 = ones(3) .* 1u"m/s"
vdf(𝐯)
```

Strip units from a distribution using `ustrip`.

```@example vdf
vdf_unitless = ustrip(vdf)
vdf_unitless(ustrip(𝐯))
```

### Visualization

The following examples demonstrate sampling from distributions and comparing with theoretical PDFs using Makie.

```@example vdf
using CairoMakie
using LinearAlgebra: norm

# Create Maxwellian distribution
vdf = Maxwellian(1.0)

# Generate samples
n_samples = 100000
vs = rand(vdf, n_samples)
speeds = norm.(vs)

# Theoretical speed distribution: f(v) = 4π v² f₃D(v)
# where f₃D is the 3D Maxwellian
v_range = range(0, 4, length=200)
speed_pdf = [4π * v^2 * vdf([v, 0, 0]) for v in v_range]

# Plot
let fig = Figure()
    ax = Axis(fig[1, 1],
        xlabel="Speed",
        ylabel="Probability Density",
        title="Maxwellian Speed Distribution")

    hist!(ax, speeds, bins=100, normalization=:pdf, label="Samples")
    lines!(ax, v_range, speed_pdf, color=:red, linewidth=2, label="Theory")
    axislegend(ax, position=:rt)
    fig
end
```

## BiMaxwellian Distribution

The BiMaxwellian distribution has different thermal velocities parallel and perpendicular to a magnetic field:

```@example vdf
vdf = BiMaxwellian(0.5, 2.0)

vs = rand(vdf, 10000)
v_pars = getindex.(vs, 3)
v_perps = [sqrt(v[1]^2 + v[2]^2) for v in vs]

v_range = range(-4, 4, length=200)
v_par_pdf = vdf.(VPar.(v_range))
v_perp_range = range(0, 4, length=200)
v_perp_pdf = vdf.(VPerp.(v_perp_range))

let fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"v_\parallel", ylabel="Probability Density", title="Bi-Maxwellian Parallel")

    hist!(ax, v_pars, bins=100, normalization=:pdf, label="Samples")
    lines!(ax, v_range, v_par_pdf, color=:red, linewidth=2, label="Theory")

    ax2 = Axis(fig[1, 2], xlabel=L"|\mathbf{v}_\perp|", title="Bi-Maxwellian Perpendicular")
    hist!(ax2, v_perps, bins=100, normalization=:pdf, label="Samples")
    lines!(ax2, v_perp_range, v_perp_pdf, color=:red, linewidth=2, label="Theory")
    axislegend(ax2, position=:rt)
    fig
end
```
