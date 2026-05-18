# Regenerate `elfin_epdef_20201001_t1.jld2` from the upstream ELFIN A L2 EPDEF CDF.

using CDFDatasets, JLD2, Downloads

const URL = "https://data.elfin.ucla.edu/ela/l2/epd/electron/2020/ela_l2_epdef_20201001_v01.cdf"
const CDF = joinpath(@__DIR__, "ela_l2_epdef_20201001_v01.cdf")
const OUT = joinpath(@__DIR__, "elfin_epdef_20201001_t1.jld2")

isfile(CDF) || Downloads.download(URL, CDF)

ds = CDFDataset(CDF)
spec = ds["ela_pef_hs_Epat_nflux"]                       # (pa × E × t)
S = Array(spec)[:, :, 1:1]
pa = Array(CDFDatasets.dim(spec, 1))[:, 1:1]              # (pa × t)
LC = Array(ds["ela_pef_hs_LCdeg"])[1:1]                   # (t,)

JLD2.save(OUT, Dict("S" => S, "pa" => pa, "LC" => LC))
@info "Wrote fixture" path = OUT size_KB = round(filesize(OUT) / 1024, digits = 2)
