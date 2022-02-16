using MLDatasets, Plots, Images, TestImages
include("src/image_views.jl")
include("src/data_utils/DataProcessing.jl")
include("src/solvers/activation_functions.jl")
include("src/solvers/cost_functions.jl")
include("src/solvers/Solvers.jl")
include("src/net.jl")



"
train_x, train_y = Fashion_MNIST.traindata()
test_x,  test_y  = Fashion_MNIST.testdata()
"

train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()


train_data = DataProcessing.prep(train_x, train_y)
test_data = DataProcessing.prep(test_x, test_y)

# Initialize weights and bias with one input layer, two hidden layers, and one 
# output layer 
W, b  = initialize_net(784, [70, 70], 10) 

# Fit weights and biases to the training data
Solvers.random_SGD!(W, b, train_data, 1_000, 0.0124, show_cost = true)
plot!()

