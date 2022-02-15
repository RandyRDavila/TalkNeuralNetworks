module DataProcessing

export prep

function prep(feature_tensor, label_vector)
    # Flatten the matrix input data into a vector
    X = []   # Flattened 784 column vectors
    Y = []   # One-hot encoding label vectors 
    (m, n, z) = size(feature_tensor)
    for i = 1:z
        push!(X, reshape(feature_tensor[:,:,i], m*n))
        y = zeros(10)
        y[label_vector[i] + 1] = 1.0
        push!(Y,y)
    end
    return [x for x in zip(X, Y)]
end

end