using VelocityDistributionFunctions
using Documenter

DocMeta.setdocmeta!(VelocityDistributionFunctions, :DocTestSetup, :(using VelocityDistributionFunctions); recursive=true)

makedocs(;
    modules=[VelocityDistributionFunctions],
    authors="Beforerr <zzj956959688@gmail.com> and contributors",
    sitename="VelocityDistributionFunctions.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaSpacePhysics.github.io/VelocityDistributionFunctions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaSpacePhysics/VelocityDistributionFunctions.jl",
    devbranch="main",
)
