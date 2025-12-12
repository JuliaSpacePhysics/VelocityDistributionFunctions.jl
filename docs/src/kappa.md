# Kappa Distribution

The Kappa distribution has power-law tails and is commonly observed in space plasmas. Its 3D PDF $f(\mathbf{v})$ is:

$$
f(\mathbf{v}) = A_3 \left(1 + \frac{|\mathbf{v} - \mathbf{u}_0|^2}{\kappa v_{th}^2}\right)^{-(\kappa + 1)}$$

where the 3D normalization constant $A_3$ is:

$$A_3 = \frac{\Gamma(\kappa + 1)}{(\pi \kappa v_{th}^2)^{3/2} \Gamma(\kappa - 1/2)}$$

```@example kappa
using VelocityDistributionFunctions
import VelocityDistributionFunctions as VDFs
using VelocityDistributionFunctions: V
using Random, Unitful, LinearAlgebra
using CairoMakie

# Create Kappa distribution: vth=1.0, κ=3.0
vdf = Kappa(1.0, 3.0)

# Create Kappa distribution with temperature
T = 30000u"K"
vdf2 = Kappa(T, 4.0)
𝐯 = ones(3) .* 1u"m/s"
# Get the PDF value at 𝐯
vdf2(𝐯)
```

## Example: Compare 1D projection PDF with theory

```@example kappa
Random.seed!(123)
vs = rand(vdf, 10000)
vxs = getindex.(vs, 1)
v_range = range(-8, 8, length=200)
vx_pdf = VDFs._pdf_1d.(vdf, v_range)

let fig = Figure()
    ax = Axis(fig[1, 1], title = "Kappa (kappa=3)", xlabel = "vx", ylabel = "PDF", yscale = log10)
    hist!(ax, vxs, normalization = :pdf, bins = 48, label = "Sampled")
    lines!(ax, v_range, vx_pdf, label = "Theory", color = :red, linewidth = 2)
    axislegend(ax)
    ylims!(ax, 3e-4, nothing)
    fig
end
```

## Example: Sampling Kappa Heavy Tails

```@example kappa
kappas = [2.5, 5.0, 10.0, 100.0]
colors = [:red, :orange, :green, :blue]

fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1],
    xlabel="Speed",
    ylabel="Probability Density",
    title="Kappa Distribution: Effect of κ parameter",
    yscale=log10)

vmax = 10
v_range = range(0, vmax, length=200)

for (κ, color) in zip(kappas, colors)
    d = Kappa(1.0, κ)

    # Generate samples
    n_samples = 100000
    samples = rand(d, n_samples)
    speeds = norm.(samples)

    # Theoretical PDF
    speed_pdf = [4π * v^2 * d([v, 0, 0]) for v in v_range]

    # Plot
    hist!(ax, speeds, bins=100, normalization=:pdf,
          color=(color, 0.3), label="κ=$κ (samples)")
    lines!(ax, v_range, speed_pdf, color=color, linewidth=2,
           label="κ=$κ (theory)", linestyle=:dash)
end

axislegend(ax, position=:rt)
xlims!(ax, 0, vmax)
ylims!(ax, 1e-4, nothing)
fig
```

## Example: Comparing Maxwellian and Kappa Distributions

```@example kappa
d_max = Maxwellian(1.0)
κ = 3.0
d_kappa = Kappa(1.0, κ)

n_samples = 50000
samples_max = rand(d_max, n_samples)
samples_kappa = rand(d_kappa, n_samples)

speeds_max = norm.(samples_max)
speeds_kappa = norm.(samples_kappa)

# Theoretical distributions
v_range = range(0, 5, length=200)
speed_pdf_max = d_max.(V.(v_range))
speed_pdf_kappa = d_kappa.(V.(v_range))

# Plot
let fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], xlabel="Speed", ylabel="Probability Density", title="Maxwellian vs Kappa Distribution (κ=$κ)", yscale=log10)
    hist!(ax, speeds_max, bins=100, normalization=:pdf, label="Maxwellian (samples)", alpha=0.5)
    hist!(ax, speeds_kappa, bins=100, normalization=:pdf, label="Kappa (samples)", alpha=0.5)
    lines!(ax, v_range, speed_pdf_max, color=:blue, linewidth=2, label="Maxwellian (theory)", linestyle=:dash)
    lines!(ax, v_range, speed_pdf_kappa, color=:red, linewidth=2, label="Kappa (theory)", linestyle=:dash)
    axislegend(ax, position=:rt)
    ylims!(ax, 1e-4, nothing)
    fig
end
```

## Math notes

### The Theoretical 1D PDF

The 1D projection of the given 3D Kappa distribution is:

$$f_{1D}(v_x) = \frac{\Gamma(\kappa)}{\sqrt{\pi \kappa} v_{th} \Gamma(\kappa - 1/2)} \left(1 + \frac{(v_x - u_{0x})^2}{\kappa v_{th}^2}\right)^{-\kappa}$$

Notes that the power changes from $-(\kappa + 1)$ in 3D to $-\kappa$ in 1D.

#### Mathematical Derivation

To find the 1D PDF $f(v_x)$, we must integrate out the other velocity components ($v_y$ and $v_z$). Let $\mathbf{v} - \mathbf{u}_0 = (u_x, u_y, u_z)$. We integrate over $u_y$ and $u_z$:

$$f(v_x) = \iint_{-\infty}^{\infty} A_3 \left(1 + \frac{u_x^2 + u_y^2 + u_z^2}{\kappa v_{th}^2}\right)^{-(\kappa + 1)} du_y \, du_z$$

After grouping the $x$ terms into a constant $C$ (relative to the integration variables $u_y, u_z$):
$$C = 1 + \frac{u_x^2}{\kappa v_{th}^2}$$

The integrand becomes:
$$\left(C + \frac{u_y^2 + u_z^2}{\kappa v_{th}^2}\right)^{-(\kappa + 1)} = C^{-(\kappa+1)} \left(1 + \frac{u_y^2 + u_z^2}{C \kappa v_{th}^2}\right)^{-(\kappa + 1)}$$

**1. Perform the Integration**

We convert the $u_y, u_z$ plane to polar coordinates ($r, \theta$) where $r^2 = u_y^2 + u_z^2$. The area element is $2\pi r dr$.
Let $a^2 = C \kappa v_{th}^2$.

$$I = \int_{0}^{\infty} \left(1 + \frac{r^2}{a^2}\right)^{-(\kappa + 1)} 2\pi r \, dr$$

Using the substitution $w = 1 + \frac{r^2}{a^2}$, we get $dw = \frac{2r}{a^2} dr \implies 2r dr = a^2 dw$.

$$I = \pi a^2 \int_{1}^{\infty} w^{-(\kappa + 1)} \, dw = \pi a^2 \left[ \frac{w^{-\kappa}}{-\kappa} \right]_{1}^{\infty} = \frac{\pi a^2}{\kappa}$$

Substitute $a^2 = C \kappa v_{th}^2$ back:

$$I = \frac{\pi (C \kappa v_{th}^2)}{\kappa} = \pi v_{th}^2 C$$

**3. Combine and Simplify Constants**

Now substitute the integral result $I$ and the constant term $C^{-(\kappa+1)}$ back into the expression for $f(v_x)$:

$$f(v_x) = A_3 \cdot C^{-(\kappa+1)} \cdot I = A_3 \pi v_{th}^2 C^{-\kappa}$$

Now expand $A_3$ and simplify:

$$f(v_x) = \left[ \frac{\Gamma(\kappa + 1)}{(\pi \kappa v_{th}^2)^{3/2} \Gamma(\kappa - 1/2)} \right] \pi v_{th}^2 \left(1 + \frac{u_x^2}{\kappa v_{th}^2}\right)^{-\kappa}$$

Isolate the constants:

$$\text{Coeff} = \frac{\Gamma(\kappa + 1) \pi v_{th}^2}{\pi^{3/2} \kappa^{3/2} v_{th}^3 \Gamma(\kappa - 1/2)} = \frac{\Gamma(\kappa + 1)}{\sqrt{\pi} \kappa^{3/2} v_{th} \Gamma(\kappa - 1/2)}$$

Using the property $\Gamma(\kappa + 1) = \kappa \Gamma(\kappa)$, we cancel one $\kappa$:

$$\text{Coeff} = \frac{\kappa \Gamma(\kappa)}{\sqrt{\pi} \kappa \sqrt{\kappa} v_{th} \Gamma(\kappa - 1/2)} = \frac{\Gamma(\kappa)}{\sqrt{\pi \kappa} v_{th} \Gamma(\kappa - 1/2)}$$

This yields the final result.
