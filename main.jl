# Modules
using MLDatasets, Random

# Reproducibility
Random.seed!(1)

# Hyperparameters
hparam = alpha, iters, h_size, data_size, n_labels = 0.005, 350, 40, 1000, 10

function main(hparam)
    # Parse hparam
    alpha, iters, net_shape, data_size, n_labels = hparam

    # Get data
    # The load_data function cleans it up too
    train_x, train_y, test_x, test_y = load_data(data_size)

    train_data = train_x, train_y

    test_data = test_x, test_y
    # Define parameters for use(ppi: pixels per image)
    ppi = reduce(*, size(xtrain[1]))

    # get_weights function either loads data or initializes random weights
    weights = get_weights(ppi, net_shape)

    # trainweight! function changes weights value
    trainweights!(weights, alpha, iters, train_data)

    nothing
end

function load_data(data_size)
    # Load data from MLDatasets module
    train_x, train_y = MNIST.traindata()
    test_x, test_y = MNIST.testdata()

    # Reshape training and test input data
    # Note that we choose a subset of the actual data, of which its size is given in hparam
    train_x = reshape(train_x[1:data_size], 1)
    test_x = reshape(test_x, 1)

    # Turn output labels to one hot vectors
    # Note that as above, we choose a subset of the output labels.
    train_y = one_hot_vectorize(train_y[1:data_size])
    test_y = one_hot_vectorize(test_y)

    train_x, train_y, test_x, test_y
end

function one_hot_vectorize(labels)
    # Initialize
    one_hot_labels = zeros(length(labels), n_labels)

    # Fill
    for (i, l) in enumerate(labels)
        one_hot_labels[i, l] = 1
    end
    one_hot_labels
end

function get_weights(input_size, net_shape, prev=false)
    if prev == true
    end

    # Initialize weights , which is a vector of arrays
    weights = Vector[]
    
    # Initial condition
    weight = init_weight(input_size, net_shape[1])
    push!(weights, weight)

    for i in 2:length(net_shape)
        weight = init_weight(net_shape[i-1], net_shape[i])
        push!(weights, weight)
    end

    weights
end

function init_weight(prev_layer_size, current_layer_size)
    0.2 * rand((prev_layer_size, current_layer_size)) - 0.1
end

function trainweight!(weights, alpha, iters, train_data)
    # Parse train_data
    train_x, train_y = train_data

    for j = 1:iters
        error, correct_cnt = 0.0, 0
        for i = 1:length(train_x)
            cache = forward(inputs[i], weights)
            error += sum(labels[i] - cache[end])
            correct_cnt += Int(argmax(cache[end])==argmax(labels[i]))
            backward!(weights, alpha, cache, train_y)
        end

        print(j, error, correct_cnt)
    end
end

# TODO: forward, backward!