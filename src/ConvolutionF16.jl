module ConvolutionF16

# Write your package code here.
using CUDA
using ChainRulesCore

export conv_depthwise_1D, conv_1D, 
       conv_depthwise_2D, conv_2D, 
       crosscor_depthwise_1D, crosscor_1D,
       crosscor_depthwise_2D, crosscor_2D

include("basic.jl")
include("backproprules.jl")

end
