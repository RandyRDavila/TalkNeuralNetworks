module NetworkMethods

export forward_pass, predict, deltas

include("activation_functions.jl")


# Define feedforward pass of the network 
function forward_pass(W, b, x)
    Z = [[0.0]]
    A = [x[1]]
    for i = 2:size(W)[1]
        push!(Z, W[i]*A[i-1] + b[i])
        push!(A, σ.(Z[i]))
    end
    return Z, A
end

# Define the predicted outcome of the network with a function
function predict(W, b, x)
    Z, A = forward_pass(W, b, x)
    return argmax(A[end])- 1
end


function deltas(W, b, x) 
    Z, A = forward_pass(W, b, x)
    L = size(W)[1]
    δ = Dict()
    δ[L] = (A[end] - x[2]).*dσ.(Z[end])
    for i = L-1:-1:2
        δ[i] = (W[i+1]'*δ[i+1]).*dσ.(Z[i])
    end
    return A, δ
end

end