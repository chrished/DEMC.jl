using Documenter, DEMC

makedocs(;
    modules=[DEMC],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chrished/DEMC.jl/blob/{commit}{path}#L{line}",
    sitename="DEMC.jl",
    authors="christoph hedtrich"
)

deploydocs(;
    repo="github.com/chrished/DEMC.jl",
)
