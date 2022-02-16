
# Initialize your weights/bias according to a normalized distribution
# with mean = 0.0 and standard deviation 1.0
function initialize_net(input_layer_size, 
                        hidden_layer_sizes, 
                        output_layer_size)

    W = [[0.0], randn(hidden_layer_sizes[1], input_layer_size)]
    b = [[0.0], randn(hidden_layer_sizes[1])]
    
    for i = 2:size(hidden_layer_sizes)[1]
        push!(W, randn(hidden_layer_sizes[i], hidden_layer_sizes[i-1]))
        push!(b, randn(hidden_layer_sizes[i]))
    end
    
    push!(W, randn(output_layer_size, hidden_layer_sizes[end]))
    push!(b, randn(output_layer_size))
    
    return W, b
end

