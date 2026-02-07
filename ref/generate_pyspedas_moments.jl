"""
    generate_pyspedas_reference.jl

Run PySPEDAS's MMS FPI ion burst moments test, extract both the reference moments
and the cleaned per-timestep distribution data, then serialize everything to
`test/refdata/pyspedas_mms_fpi_brst_i.jls` so that unit tests can load it
without touching Python.

Usage (from project root):
    julia --project=test test/generate_pyspedas_reference.jl
"""

using PySPEDAS
using PySPEDAS.PythonCall
using Serialization
using Chairmarks

@py import pyspedas.projects.mms.tests.test_mms_part_getspec: PGSTests
@py import pyspedas.projects.mms.fpi_tools.mms_get_fpi_dist: mms_get_fpi_dist
@py import pyspedas.projects.mms.particles.mms_convert_flux_units: mms_convert_flux_units
@py import pyspedas.projects.mms.particles.mms_pgs_clean_data: mms_pgs_clean_data
@py import pyspedas.projects.mms.particles.mms_pgs_clean_support: mms_pgs_clean_support
@py import pyspedas.tplot_tools: get_data as py_get_data
@py import numpy as np
@py import pyspedas.projects.mms.particles.mms_part_getspec: mms_part_getspec
@py import pyspedas.particles.moments.spd_pgs_moments: spd_pgs_moments

# ── 1. Run pyspedas to generate reference moments ──────────────────────
println("Running PySPEDAS mms_part_getspec …")
tnames = mms_part_getspec(
    trange = ["2015-10-16/13:06:00", "2015-10-16/13:06:10"],
    data_rate = "brst",
    species = "i",
    output = "moments",
    prefix = "pre_"
)
println("  done.")

tname = "mms1_dis_dist_brst"

# ── 2. Extract reference moment arrays ──────────────────────────────────────
ref = Dict{String, Any}()
for name in [
        "density", "flux", "velocity", "eflux", "qflux",
        "mftens", "ptens", "ttens", "avgtemp", "vthermal",
        "t3", "magt3", "symm", "symm_theta", "symm_phi", "symm_ang",
    ]
    tvar = "pre_mms1_dis_dist_brst_$name"
    d = py_get_data(tvar)
    if pyconvert(Bool, d == pybuiltins.None)
        @warn "tplot variable $tvar not found, skipping"
        continue
    end
    ref[name] = pyconvert(Array{Float64}, d.y)
end

# ── 3. Extract cleaned distribution data per timestep ───────────────────────
data_in = py_get_data(tname)
times = pyconvert(Vector{Float64}, data_in.times)
ntimes = length(times)
println("Number of timesteps: $ntimes")

support = mms_pgs_clean_support(
    np.array(times);
    mag_name = "mms1_fgm_b_gse_brst_l2_bvec",
    sc_pot_name = "mms1_edp_scpot_brst_l2"
)
mag_data = pyconvert(Matrix{Float64}, support[0])   # (ntimes, 3)
scpot_data = pyconvert(Vector{Float64}, support[2])    # (ntimes,)

dists_jl = Vector{Dict{String, Any}}(undef, ntimes)

btime = 0.

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
    clean["magf"] = support[0][i-1]

    btime += (@b spd_pgs_moments(clean, sc_pot = scpot_data[i])).time

    dists_jl[i] = Dict{String, Any}(
        "data" => pyconvert(Matrix{Float64}, clean["data"]),
        "energy" => pyconvert(Matrix{Float64}, clean["energy"]),
        "denergy" => pyconvert(Matrix{Float64}, clean["denergy"]),
        "theta" => pyconvert(Matrix{Float64}, clean["theta"]),
        "dtheta" => pyconvert(Matrix{Float64}, clean["dtheta"]),
        "phi" => pyconvert(Matrix{Float64}, clean["phi"]),
        "dphi" => pyconvert(Matrix{Float64}, clean["dphi"]),
        "bins" => pyconvert(Matrix{Float64}, clean["bins"]),
        "mass" => pyconvert(Float64, clean["mass"]),
        "charge" => pyconvert(Float64, clean["charge"]),
    )
end
@info "Total time: $(btime) s"
println("\n  done.")

# ── 4. Save ─────────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "refdata")
mkpath(outdir)
outfile = joinpath(outdir, "pyspedas_mms_fpi_brst_i.jls")

result = Dict{String, Any}(
    "ref_moments" => ref,
    "distributions" => dists_jl,
    "mag_data" => mag_data,
    "scpot_data" => scpot_data,
    "times" => times,
)

serialize(outfile, result)
println("Saved reference data to $outfile ($(filesize(outfile) ÷ 1024) KB)")
