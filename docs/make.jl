using VelocityDistributionFunctions
using Documenter
using DocumenterCitations

const bib = CitationBibliography(joinpath(@__DIR__, "VelocityDistributionFunctions.jl.bib"))

DocMeta.setdocmeta!(VelocityDistributionFunctions, :DocTestSetup, :(using VelocityDistributionFunctions); recursive = true)

makedocs(;
    modules = [VelocityDistributionFunctions],
    authors = "Beforerr <zzj956959688@gmail.com> and contributors",
    sitename = "VelocityDistributionFunctions.jl",
    format = Documenter.HTML(;
        canonical = "https://JuliaSpacePhysics.github.io/VelocityDistributionFunctions.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Distributions" => "distributions.md",
        "Kappa" => "kappa.md",
        "Math notes for moment calculations" => "moment_note.md",
    ],
    plugins = [bib],
)

deploydocs(;
    repo = "github.com/JuliaSpacePhysics/VelocityDistributionFunctions.jl",
    push_preview = true,
)
