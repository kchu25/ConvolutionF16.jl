using ConvolutionF16
using Documenter

DocMeta.setdocmeta!(ConvolutionF16, :DocTestSetup, :(using ConvolutionF16); recursive=true)

makedocs(;
    modules=[ConvolutionF16],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    sitename="ConvolutionF16.jl",
    format=Documenter.HTML(;
        canonical="https://kchu25.github.io/ConvolutionF16.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/ConvolutionF16.jl",
    devbranch="main",
)
