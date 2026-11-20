# Shared ELFIN EPD constants, parameter helpers, and fixture loader.
# See `fixtures/gen_elfin_fixture.jl` for regeneration recipe.

using JLD2

# Energy bin widths in keV: `EPD_ENERGY_BINS_MAX .- EPD_ENERGY_BINS_MIN`.
const ELFIN_EPD_dE = Float32[30, 40, 40, 50, 60, 75, 85, 200, 270, 400, 500, 700, 850, 800, 1650, 2200]

# Default channel groupings used by `epd_l2_PAspectra`.
const ELFIN_PA_CHANNELS = [1:3, 4:6, 7:9, 10:16]

const ELFIN_FIXTURE_PATH = joinpath(@__DIR__, "elfin_epdef_20201001_t1.jld2")

# Mirrors `ELFINData.epd_l2_Espectra` parameter derivation for half-spin EPD data.
function elfin_des_params(n_pa; fullspin = false)
    nspinsectors = fullspin ? (n_pa - 2) : (n_pa - 2) * 2
    FOVo2 = 11.0
    half_sector_width = 180 / nspinsectors
    para_tol = FOVo2 + half_sector_width
    perp_tol = -FOVo2
    return (; half_sector_width, para_tol, perp_tol)
end

"""
    load_elfin_fixture(; n_t = 1)

Load the bundled ELFIN-A L2 EPDEF fixture (single time step) and optionally tile
along the time axis to `n_t` records. Returns `(; S, pa, LC)` with shapes
`(n_pa, n_E, n_t)`, `(n_pa, n_t)`, and `(n_t,)` respectively.
"""
function load_elfin_fixture(; n_t::Integer = 1)
    data = JLD2.load(ELFIN_FIXTURE_PATH)
    n_t == 1 && return (; S = data["S"], pa = data["pa"], LC = data["LC"])
    return (;
        S = repeat(data["S"], 1, 1, n_t),
        pa = repeat(data["pa"], 1, n_t),
        LC = repeat(data["LC"], n_t),
    )
end
