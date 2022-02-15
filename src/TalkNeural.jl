module TalkNeural

include("net.jl")
export initialize_net

include("solvers/Solvers.jl")
using .Solvers
export SGD!, random_SGD!


include("data_utils/DataProcessing.jl")
using .DataProcessing
export prep

end # module