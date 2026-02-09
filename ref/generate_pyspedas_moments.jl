"""
    generate_pyspedas_reference.jl

Generate spectra and moments from 3D MMS particle data.

Extract both the reference moments and the cleaned per-timestep distribution data, then serialize it so that unit tests can load it without touching Python.

Usage (from project root):
    julia --project=test test/generate_pyspedas_reference.jl

## References

- [mms_part_getspec](https://github.com/spedas/pyspedas/blob/master/pyspedas/projects/mms/particles/mms_part_getspec.py)
- [mms_part_products](https://github.com/spedas/pyspedas/blob/master/pyspedas/projects/mms/particles/mms_part_products.py)
- [mms_get_fpi_dist](https://github.com/spedas/pyspedas/blob/master/pyspedas/projects/mms/fpi_tools/mms_get_fpi_dist.py)
- [mms_pgs_clean_data](https://github.com/spedas/pyspedas/blob/master/pyspedas/projects/mms/particles/mms_pgs_clean_data.py)
"""

using PySPEDAS
using PySPEDAS.Projects
using PySPEDAS.PythonCall
using JLD2
using Chairmarks

@py import pyspedas.projects.mms.fpi_tools.mms_get_fpi_dist: mms_get_fpi_dist
@py import pyspedas.projects.mms.particles.mms_convert_flux_units: mms_convert_flux_units
@py import pyspedas.projects.mms.particles.mms_pgs_clean_data: mms_pgs_clean_data
@py import pyspedas.projects.mms.particles.mms_pgs_clean_support: mms_pgs_clean_support
@py import pyspedas.tplot_tools: get_data as py_get_data
@py import pyspedas.projects.mms.particles.mms_part_getspec: mms_part_getspec
@py import pyspedas.particles.moments.spd_pgs_moments: spd_pgs_moments

# ── 1. Run pyspedas to generate reference moments ──────────────────────
@info "Running PySPEDAS mms_part_getspec …"
trange = ["2015-10-16/13:06:00", "2015-10-16/13:06:10"]
tnames = mms_part_getspec(
    trange = trange,
    data_rate = "brst", species = "i",
    output = ["moments", "theta", "energy"],
    prefix = "pre_"
)

tname = "mms1_dis_dist_brst"
suffix = "pre_mms1_dis_dist_brst_"

# ── 2. Extract reference moment arrays ──────────────────────────────────────
names = [
    "density", "flux", "velocity", "eflux", "qflux",
    "mftens", "ptens", "ttens", "avgtemp", "vthermal",
    "t3", "magt3", "symm", "symm_theta", "symm_phi", "symm_ang",
    "energy", "theta",
]
ref = Dict(name => Array(get_data(suffix * name)) for name in names)


# ── 3. Extract cleaned distribution data per timestep ───────────────────────
data_in = py_get_data(tname)

# Save the data for unit tests
dis_dist_brst = Dict(
    "data" => pyconvert(Array, data_in.y),
    "phi" => pyconvert(Array, data_in.v1),
    "theta" => pyconvert(Array, data_in.v2),
    "energy" => pyconvert(Array, data_in.v3),
)


py_times = data_in.times
times = pyconvert(Vector{Float64}, py_times)
ntimes = length(times)
println("Number of timesteps: $ntimes")

support = mms_pgs_clean_support(
    py_times;
    mag_name = "mms1_fgm_b_gse_brst_l2_bvec",
    sc_pot_name = "mms1_edp_scpot_brst_l2"
)
mag_data = pyconvert(Matrix{Float64}, support[0])   # (ntimes, 3)
scpot_data = pyconvert(Vector{Float64}, support[2])    # (ntimes,)

dists_jl = Vector{Dict{String, Any}}(undef, ntimes)

btime = 0.0

for i in 1:ntimes
    print("\r  extracting distribution $i / $ntimes")
    dists_py = mms_get_fpi_dist(
        tname; index = i - 1,
        species = "i", probe = "1", data_rate = "brst"
    )
    dist_dict = dists_py[0]

    # orig_energy is needed by clean_data
    dist_dict["orig_energy"] = dist_dict["energy"][pybuiltins.slice(pybuiltins.None), 0, 0]

    eflux_data = mms_convert_flux_units(dist_dict; units = "eflux")
    clean = mms_pgs_clean_data(eflux_data)
    clean["magf"] = support[0][i - 1]

    btime += (@b spd_pgs_moments(clean, sc_pot = scpot_data[i])).time

    dists_jl[i] = Dict{String, Any}(
        "data" => pyconvert(Matrix, clean["data"]),
        "energy" => pyconvert(Matrix, clean["energy"]),
        "denergy" => pyconvert(Matrix, clean["denergy"]),
        "theta" => pyconvert(Matrix, clean["theta"]),
        "dtheta" => pyconvert(Matrix, clean["dtheta"]),
        "phi" => pyconvert(Matrix, clean["phi"]),
        "dphi" => pyconvert(Matrix, clean["dphi"]),
    )
end
@info "Total time: $(btime) s"
println("\n  done.")

# ── 4. Save ─────────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "../data")
mkpath(outdir)
outfile = joinpath(outdir, "pyspedas_mms_fpi_brst_i.jld2")

result = Dict{String, Any}(
    "ref" => ref,
    "dis_dist_brst" => dis_dist_brst,
    "distributions" => dists_jl,
    "mag_data" => mag_data,
    "scpot_data" => scpot_data,
    "times" => times,
)

jldsave(outfile; result)
println("Saved reference data to $outfile ($(filesize(outfile) ÷ 1024) KB)")


## Benchmarking
@b for i in 1:ntimes
    dists_py = mms_get_fpi_dist(
        tname; index = i - 1,
        species = "i", probe = "1", data_rate = "brst"
    )
    dist_dict = dists_py[0]
    # orig_energy is needed by clean_data
    dist_dict["orig_energy"] = dist_dict["energy"][pybuiltins.slice(pybuiltins.None), 0, 0]
    eflux_data = mms_convert_flux_units(dist_dict; units = "eflux")
    mms_pgs_clean_data(eflux_data)
end
