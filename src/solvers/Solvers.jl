module Solvers

using Plots
include("activation_functions.jl")
include("cost_functions.jl")
include("NetworkMethods.jl")


export SGD!, random_SGD!

function SGD!(W, b, data_set, epochs, η;  batch_size = 1, show_cost = false)
    L = size(W)[1]
    N = length(data_set)
    if show_cost == true
        #cost_points = [(0, total_MSE(W_new, b_new, data_set))]
        errors_ = []
        for j = 1:epochs
            errors = 0.0
            for d in data_set
                A, δ = NetworkMethods.deltas(W, b, d)
                for i = L:-1:2
                    W[i] -= (η/batch_size)*δ[i]*A[i-1]'
                    b[i] -= (η/batch_size)*δ[i]
                end
                errors += MSE(A[end], d[1]) 
            end
            push!(errors_, (j, errors))
        end
        n = η
        error_length = length(errors_)
        Plots.plot(1:error_length,
                   errors_, 
                   xaxis = "Epochs", 
                   yaxis = "Cost",
                   xticks = 1:error_length,
                   title = "epochs = $epochs, batch size = $batch_size, eta = $n",
                   legend = false)
        return W, b
    else
        for j = 1:epochs
            for d in data_set
                A, δ = NetworkMethods.deltas(W, b, d)
                for i = L:-1:2
                    W[i] -= (η/batch_size)*δ[i]*A[i-1]'
                    b[i] -= (η/batch_size)*δ[i]
                end 
            end
        end

        return W, b
    end
end

function random_SGD!(W, b, data_set, epochs, η;  batch_size = 1, show_cost = false)
    L = size(W)[1]
    N = length(data_set)
    
    if show_cost == true
        cost_points = [(0, total_MSE(W, b, data_set))]

        for j = 1:epochs
            k = rand(1:size(data_set)[1]-batch_size)
            batch = data_set[k:k+batch_size]
            for x in batch
                A, δ = NetworkMethods.deltas(W, b, x)
                for i = L:-1:2
                    W[i] -= (η/batch_size)*δ[i]*A[i-1]'
                    b[i] -= (η/batch_size)*δ[i]
                end 
            end
        
        push!(cost_points, (j, total_MSE(W, b, data_set)))
        end
        n = η
        plot(cost_points, 
             xaxis = "Epochs", 
             yaxis = "Cost",
             xticks = 1:length(cost_points),
             title = "epochs = $epochs, batch size = $batch_size, eta = $n",
             legend = false)
        
        return W, b
    else
        for j = 1:epochs
            k = rand(1:size(data_set)[1]-batch_size)
            batch = data_set[k:k+batch_size]
            for x in batch
                A, δ = NetworkMethods.deltas(W, b, x)
                for i = L:-1:2
                    W[i] -= (η/batch_size)*δ[i]*A[i-1]'
                    b[i] -= (η/batch_size)*δ[i]
                end 
            end
        end

        return W, b
    end
end

end