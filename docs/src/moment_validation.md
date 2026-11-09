# Moments validation with PySPEDAS

Retrieve the data from PySPEDAS into Julia, and compute the moments directly
from the raw distribution — no per-timestep `mms_get_fpi_dist` calls needed.

Here we load the data from a JLD2 file in order to avoid the slow PySPEDAS
run.

```@example moment
using VelocityDistributionFunctions
using Downloads
using JLD2

function download_test_data(url, filename = basename(url))
    cache_dir = joinpath(pkgdir(VelocityDistributionFunctions), "data")
    mkpath(cache_dir)
    filepath = joinpath(cache_dir, filename)
    if !isfile(filepath)
        @info "Downloading test data: $filename"
        Downloads.download(url, filepath)
    end
    return filepath
end

const ref_url = "https://github.com/JuliaSpacePhysics/VelocityDistributionFunctions.jl/releases/download/v0.2.1/pyspedas_mms_fpi_brst_i.jld2"

reffile = download_test_data(ref_url)
saved = JLD2.load(reffile)["result"]
```

MMS FPI angles are instrument look directions; the particle velocity direction is `θ → 180 - θ`, `φ → φ + 180`.

```@example moment
mms1_dis_dist_brst = saved["dis_dist_brst"]
data = mms1_dis_dist_brst["data"]
theta = 180 .- mms1_dis_dist_brst["theta"]
phi = (mms1_dis_dist_brst["phi"] .+ 180) .% 360
energy = mms1_dis_dist_brst["energy"]
scpot_data = saved["scpot_data"]
out = tmoments(data, theta, phi, energy, scpot_data; edim=4, tdim=1, units=:df_cm, species=:H)
```