using ConvolutionF16
using Flux, CUDA
using Test

@testset "ConvolutionF16.jl" begin
    # Write your tests here.
    n_grps = 5

    # test conv_depthwise_1D #######################
    f = rand(3, 1, n_grps)
    x = rand(6, n_grps, 2)
    f_gpu = f |> cu
    x_gpu = x |> cu
    @test all(conv_depthwise_1D(f_gpu, x_gpu) .≈ conv(x_gpu, f_gpu, pad=2, groups=n_grps))
    # test the gradient of conv_depthwise_1D
    ps = Flux.params(f_gpu, x_gpu);
    gs_flux = gradient(ps) do
        conv(x_gpu, f_gpu, pad=2, groups=n_grps) |> sum
    end
    gs = gradient(ps) do
        conv_depthwise_1D(f_gpu, x_gpu) |> sum
    end
    @test all(gs_flux[f_gpu] .≈ gs[f_gpu])
    @test all(gs_flux[x_gpu] .≈ gs[x_gpu])
        
    # test conv_1D #######################
    x = rand(6, 1, 2)
    x_gpu = x |> cu
    @test all(conv_1D(f_gpu, x_gpu) .≈ conv(x_gpu, f_gpu, pad=2))
    # test the gradient of conv_1D
    ps = Flux.params(f_gpu, x_gpu);
    gs_flux = gradient(ps) do
        conv(x_gpu, f_gpu, pad=2) |> sum
    end
    gs = gradient(ps) do
        conv_1D(f_gpu, x_gpu) |> sum
    end
    @test all(gs_flux[f_gpu] .≈ gs[f_gpu])
    @test all(gs_flux[x_gpu] .≈ gs[x_gpu])

    # test crosscor_depthwise_1D #######################
    f = rand(3, 1, n_grps)
    x = rand(6, n_grps, 2)
    f_gpu = f |> cu
    x_gpu = x |> cu
    @test all(crosscor_depthwise_1D(f_gpu, x_gpu) .≈ conv(x_gpu, f_gpu, pad=0, groups=n_grps, flipped=true))
    # test the gradient of crosscor_depthwise_1D
    ps = Flux.params(f_gpu, x_gpu);
    gs_flux = gradient(ps) do
        conv(x_gpu, f_gpu, pad=0, groups=n_grps, flipped=true) |> sum
    end
    gs = gradient(ps) do
        crosscor_depthwise_1D(f_gpu, x_gpu) |> sum
    end
    @test all(gs_flux[f_gpu] .≈ gs[f_gpu])
    @test all(gs_flux[x_gpu] .≈ gs[x_gpu])

    # test crosscor_1D #######################
    x = rand(6, 1, 2)
    x_gpu = x |> cu
    @test all(crosscor_1D(f_gpu, x_gpu) .≈ conv(x_gpu, f_gpu, pad=0, flipped=true))
    # test the gradient of crosscor_1D
    ps = Flux.params(f_gpu, x_gpu);
    gs_flux = gradient(ps) do
        conv(x_gpu, f_gpu, pad=0, flipped=true) |> sum
    end
    gs = gradient(ps) do
        crosscor_1D(f_gpu, x_gpu) |> sum
    end
    @test all(gs_flux[f_gpu] .≈ gs[f_gpu])
    @test all(gs_flux[x_gpu] .≈ gs[x_gpu])

    # test conv_depthwise_2D #######################
    f = rand(2,2,1,n_grps)
    x = rand(4,1,n_grps,2)
    f_gpu = f |> cu
    x_gpu = x |> cu
    @test all(conv_depthwise_2D(f_gpu, x_gpu) .≈ conv(x_gpu, f_gpu, pad=1, groups=n_grps))
    # test the gradient of conv_depthwise_2D
    ps = Flux.params(f_gpu, x_gpu);
    gs_flux = gradient(ps) do
        conv(x_gpu, f_gpu, pad=1, groups=n_grps) |> sum
    end
    gs = gradient(ps) do
        conv_depthwise_2D(f_gpu, x_gpu) |> sum
    end
    @test all(gs_flux[f_gpu] .≈ gs[f_gpu])
    @test all(gs_flux[x_gpu] .≈ gs[x_gpu])

    # test conv_2D #######################
    f = rand(2,2,1,n_grps)
    x = rand(4,1,1,2)
    f_gpu = f |> cu
    x_gpu = x |> cu
    @test all(conv_2D(f_gpu, x_gpu) .≈ conv(x_gpu, f_gpu, pad=1))
    # test the gradient of conv_2D
    ps = Flux.params(f_gpu, x_gpu);
    gs_flux = gradient(ps) do
        conv(x_gpu, f_gpu, pad=1) |> sum
    end
    gs = gradient(ps) do
        conv_2D(f_gpu, x_gpu) |> sum
    end
    @test all(gs_flux[f_gpu] .≈ gs[f_gpu])
    @test all(gs_flux[x_gpu] .≈ gs[x_gpu])

    # test crosscor_depthwise_2D #######################
    f = rand(2,2,1,n_grps)
    x = rand(4,2,n_grps,3)
    f_gpu = f |> cu
    x_gpu = x |> cu
    @test all(crosscor_depthwise_2D(f_gpu, x_gpu) .≈ conv(x_gpu, f_gpu, pad=0, groups=n_grps, flipped=true))
    # test the gradient of crosscor_depthwise_2D
    ps = Flux.params(f_gpu, x_gpu);
    gs_flux = gradient(ps) do
        conv(x_gpu, f_gpu, pad=0, groups=n_grps, flipped=true) |> sum
    end
    gs = gradient(ps) do
        crosscor_depthwise_2D(f_gpu, x_gpu) |> sum
    end
    @test all(gs_flux[f_gpu] .≈ gs[f_gpu])
    @test all(gs_flux[x_gpu] .≈ gs[x_gpu])

    # test crosscor_2D #######################
    f = rand(2,2,1,n_grps)
    x = rand(4,2,1,5)
    f_gpu = f |> cu
    x_gpu = x |> cu
    @test all(crosscor_2D(f_gpu, x_gpu) .≈ conv(x_gpu, f_gpu, pad=0, flipped=true))
    # test the gradient of crosscor_2D
    ps = Flux.params(f_gpu, x_gpu);
    gs_flux = gradient(ps) do
        conv(x_gpu, f_gpu, pad=0, flipped=true) |> sum
    end
    gs = gradient(ps) do
        crosscor_2D(f_gpu, x_gpu) |> sum
    end
    @test all(gs_flux[f_gpu] .≈ gs[f_gpu])
    @test all(gs_flux[x_gpu] .≈ gs[x_gpu])

end
