#=
Written by: Jaha Actara
Created on: 2021-04-18
Last edited on: 2021-04-21
=#

# Modules
using MLDatasets

# Reproducibility
using Random
Random.seed!(1)

#=
  This block comment indicates that the codes from here on are what should be included in a module.
  This is true until the next block comment.

  I borrowed from tobiasbrodd's neural.jl from his(or her) neural-network-julia repository. Specifically, the notion of struct Network and the following.
  Thanks Tobias, but your code hasn't shed Python's husk. To be fair, his last commit was in 2018.
  I tried to make my code more verbose where needed, and more idiomatic. I also made the code so that z = Wx + b. 
=#

using Statistics

# The general structure of the code is as follows:
#   Get data, process it
#   Initialize mutable struct Network
#     The first element of layers field is the input, and the last layer is the output layer.
#     In this implementation, the term network, or net encompasses both the input layer and the output layer.
#       Thus, the input layer is layers[1], and the output layer is layers[N + 2](if there are N hidden layers).
#       There are N + 1 weights in total.
#       The i-th layer should be acted on by the i-th weight.
#       z = net.weights[i] * net.layers[i] .+ net.bias[i]
#     alpha is learning rate.
#   Train the network using forward and backward pass.
mutable struct Network
    layers::Array{Any, 1}
    weights::Array{Any, 1}
    bias::Array{Any, 1}
    alpha::Float64
end

function sigmoid(x)
    1 ./ (1 .+ exp.(-x))
end

function sigmoid2deriv(x)
    x .* (1 .- x)
end

function tanh(x)
    (exp.(x) - exp.(-x)) / (exp.(x) + exp.(-x))
end

function tanh2deriv(x)
    1 - x.^2
end

function relu(x)
    (x .>= 0) .* x
end

function relu2deriv(x)
    (x .>= 0)
end

function init_weight(next_layer_size, current_layer_size)
    0.2 * rand((next_layer_size, current_layer_size)) - 0.1
end

function init(input_size, h_sizes, output_size, alpha=0.01)
    sizes = [input_size h_sizes output_size]

    weights = []
    for i = 1:length(sizes)-1
        push!(weights, init_weight(sizes[i+1], sizes[i]))
    end

    bias = []
    for i = 1:length(sizes)-1
        push!(b, zeros((sizes[i+1], 1)))
    end

   return Network([], weights, bias, alpha)
end

function forward!(net::Network, x) # x is a single instance of data
    push!(net.layers, x)
    for i = 1:length(net.weights)
        z = net.weights[i] * net.layers[i] .+ net.bias[i]
        push!(net.layers, relu(z))
    end
    return network
end

function backward!(net::Network, x, y)
    output = net.layers[end]
    
    error = y - output
    delta = error .* relu2deriv(output)

    net.weights[end] += net.alpha * transpose(net.layers[end-1]) * delta
    net.bias[end] .+= net.alpha * mean(delta)

    w_length = length(net.weights)

    for i = w_length-1:1
        
        error = delta * net.weights[i+1]'
        delta = error .* relu2deriv(net.layers[i+1])
        net.weights[i] += net.alpha * net.layers[i]' * delta
        net.bias[i] .+= net.alpha * mean(delta)
    end

    return net
end

function train!(net::Network, train_data, iters)
    X, Y = train_data
    
    for i = 1:iters
        for j = 1:length(X)
            forward!(net, X[j])
            backward!(net, X[j], Y[j])
        end
    end
    
    return nothing
end

function test(net::Network, test_data)
    X, Y = test_data
    
    correct_cnt = 0
    for i = 1:length(X)
        forward!(net, X[i])
        output = net.layers[end]
        correct_cnt += Int(argmax(output)==argmax(Y[i]))
    end
    
    accuracy = correct_cnt/length(X)
    
    correct_cnt, accuracy
end
        

function predict!(net::Network, XP)
    forward!(net, XP)
end

#=
  This block comment indicates the end of codes that should have been included in a module, but wasn't. I'm lazy. Sue me.
  I confess that I still don't know how to factor my codes.
  Fira code would look better. I don't know if ligatures will be beneficial to Julia. Time will tell.
=#

# Hyperparameters
hparam = alpha, iters, h_size, data_size, output_size = 0.01, 350, 100, 1000, 10

function main(hparam)
    # Parse hparam
    alpha, iters, h_size, data_size, output_size = hparam

    # Get data
    # The load_data function cleans it up too
    (train_x, train_y), (test_x, test_y) = load_data(ppi, data_size)
    
    # segregate training data from test data
    train_data = train_x, train_y
    test_data = test_x, test_y
    
    # Define parameters for use(ppi: pixels per image)
    # This is done by taking the first image of training data.
    input_size = reduce(*, size(train_x[1]))
    
    # Initialize Network
    net = init(input_size, h_size, output_size, alpha=alpha)

    # train! function changes weights value
    train!(net, train_data, iters)
    
    println(test(net, test_data))

    nothing
end

function load_data(ppi, data_size)
    # Load data from MLDatasets module
    train_x, train_y = MNIST.traindata()
    test_x, test_y = MNIST.testdata()

    # Reshape training and test input data
    # Note that we choose a subset of the actual data, of which its size is given in hparam
    train_x = reshape(train_x[1:data_size], (ppi, data_size))
    test_x = reshape(test_x, (ppi,length(test_x))

    # Turn output labels to one hot vectors
    # Note that as above, we choose a subset of the output labels.
    train_y = one_hot_vectorize(n_labels, train_y[1:data_size])
    test_y = one_hot_vectorize(n_labels, test_y)

    train_x, train_y, test_x, test_y
end

function one_hot_vectorize(n_labels, labels)
    # Initialize
    one_hot_labels = zeros(n_labels, labels)

    # Fill
    for (label, index) in enumerate(labels)
        one_hot_labels[label, index] = 1
    end
    one_hot_labels
end 
