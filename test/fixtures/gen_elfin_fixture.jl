# Regenerate `elfin_epdef_20201001_t1.jld2` from the upstream ELFIN A L2 EPDEF CDF.

using CDFDatasets, JLD2, Downloads

const URL = "https://data.elfin.ucla.edu/ela/l2/epd/electron/2020/ela_l2_epdef_20201001_v01.cdf"
const CDF = joinpath(@__DIR__, "ela_l2_epdef_20201001_v01.cdf")
const OUT = joinpath(@__DIR__, "elfin_epdef_20201001_t1.jld2")

isfile(CDF) || Downloads.download(URL, CDF)

# ELFIN VALIDMAX (1e6) is below real trapped fluxes
ds = cdfopen(CDF; checks = (; validmin = nothing, validmax = nothing))
spec = ds["ela_pef_hs_Epat_nflux"]                       # (pa × E × t)
S = Array(spec)[:, :, 1:1]
pa = Array(ds["ela_pef_hs_epa_spec"])[:, 1:1]             # (pa × t), DEPEND_1 of spec
LC = Array(ds["ela_pef_hs_LCdeg"])[1:1]                   # (t,)

JLD2.save(OUT, Dict("S" => S, "pa" => pa, "LC" => LC))
@info "Wrote fixture" path = OUT size_KB = round(filesize(OUT) / 1024, digits = 2)
