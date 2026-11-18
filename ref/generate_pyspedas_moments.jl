"""
    ref/generate_pyspedas_moments.jl

Generate PySPEDAS reference moments and the raw 3D MMS-FPI distribution, magnetic field and spacecraft potential for the same scan, then serialize them so tests and docs can load them without Python.

Usage (from project root):
    julia --project=test test/ref/generate_pyspedas_moments.jl

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

@py import pyspedas.projects.mms.particles.mms_pgs_clean_support: mms_pgs_clean_support
@py import pyspedas.tplot_tools: get_data as py_get_data
@py import pyspedas.projects.mms.particles.mms_part_getspec: mms_part_getspec

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
ref = Dict{String, Any}(name => Array(get_data(suffix * name)) for name in names)


# ── 3. Extract the distribution, mag and spacecraft potential ──────
data_in = py_get_data(tname)

# Save the data for unit tests
dis_dist_brst = Dict{String, Any}(
    "data" => pyconvert(Array, data_in.y),
    "phi" => pyconvert(Array, data_in.v1),
    "theta" => pyconvert(Array, data_in.v2),
    "energy" => pyconvert(Array, data_in.v3),
)


py_times = data_in.times

support = mms_pgs_clean_support(
    py_times;
    mag_name = "mms1_fgm_b_gse_brst_l2_bvec",
    sc_pot_name = "mms1_edp_scpot_brst_l2"
)
mag_data = pyconvert(Matrix{Float64}, support[0])   # (ntimes, 3)
scpot_data = pyconvert(Vector{Float64}, support[2])    # (ntimes,)

# ── 4. Save ─────────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "../data")
mkpath(outdir)
outfile = joinpath(outdir, "pyspedas_mms_fpi_brst_i_v2.jld2")

result = Dict{String, Any}(
    "ref" => ref,
    "dis_dist_brst" => dis_dist_brst,
    "mag_data" => mag_data,
    "scpot_data" => scpot_data,
)

jldsave(outfile; result)
println("Saved reference data to $outfile ($(filesize(outfile) ÷ 1024) KB)")

