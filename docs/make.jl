using VelocityDistributionFunctions
using Documenter

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
        "Math notes for moment calculations" => "moment_note.md",
        "Moment validation" => "moment_validation.md",
    ],
)

deploydocs(;
    repo = "github.com/JuliaSpacePhysics/VelocityDistributionFunctions.jl",
    push_preview = true,
)
