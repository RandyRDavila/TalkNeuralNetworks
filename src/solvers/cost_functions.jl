include("NetworkMethods.jl")

# Define the cost of a given output
MSE(a, y) = (1/2)*sum((a[i] - y[i])^2 for i = 1:size(a)[1])

# Define the total cost of a dataset with respect to weights and bias
function total_MSE(W, b, labeled_data)
    cost = 0.0
    for i = 1:60000
        Z, A = NetworkMethods.forward_pass(W, b, labeled_data[i])
        cost += MSE(σ.(A[size(W)[1]]), labeled_data[i][2])
    end
    return cost/60000
end
