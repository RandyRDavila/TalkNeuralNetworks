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

net_W, net_b  = initialize_net(784, [70, 70], 10) 

Solvers.random_SGD!(net_W, net_b, train_data, 100, 0.0324, show_cost = true)
plot!()

