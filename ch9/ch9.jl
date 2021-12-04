# Modules
using MLDatasets, Random

# Reproducibility
Random.seed!(1)

# Hyperparameters
hparam = alpha, iters, h_size, batch_size = 2, 300, 100, 100

# Metadata
metadata = data_size, n_data, n_labels = (28,28), 1000, 10

function main(hparam, metadata)
    # Parse hparam
    alpha, iters, h_size, batch_size = hparam

    # Parse metadata
    data_size, n_data, n_labels = metadata

    # Get data
    # The load_data function cleans it up too
    train_data, test_data = load_data(data_size, n_data)

    # get_weights function either loads data or initializes random weights
    weights = get_weights(data_size, h_size, n_labels)

    # trainweight! function changes weights value
    train!(weights, hparam, metadata, train_data, test_data)
end

function load_data(data_size, n_data)
    # Parse data_size
    wid, len = data_size

    ppi = wid * len

    # Load data from MLDatasets module
    train_x, train_y = MNIST.traindata()
    test_x, test_y = MNIST.testdata()

    # Reshape training and test input data
    # Note that we choose a subset of the actual data, of which its size is given in hparam
    train_x = reshape(train_x[:, :, 1:n_data], (ppi, n_data))
    train_y = one_hot(n_labels, train_y[1:n_data])

    test_x = reshape(test_x, (ppi, size(test_x)[3]))
    test_y = one_hot(n_labels, test_y)
    
    train_data = train_x, train_y
    test_data = test_x, test_y

    return train_data, test_data
end

function one_hot(n_labels, labels)
    # Initialize
    one_hot_labels = zeros(n_labels, length(labels))

    # Fill
    for (index, label) in enumerate(labels)
        one_hot_labels[label+1, index] = 1
    end
    one_hot_labels
end

function get_weights(data_size, h_size, n_labels)
    # get pixels per image
    ppi = reduce(* , data_size)
    
    weights = Vector()

    # Initial condition
    weight = init_weight(h_size, ppi)
    push!(weights, weight)

    weight = init_weight(n_labels, h_size)
    push!(weights, weight)

    weights
end

function init_weight(next_layer_size, prev_layer_size)
    0.2 .* rand(next_layer_size, prev_layer_size) .- 0.1
end

function train!(weights, hparam, metadata, train_data, test_data)
    # Parse
    train_x, train_y = train_data
    test_x, test_y = test_data

    alpha, iters, h_size, batch_size = hparam

    data_size, n_data, n_labels = metadata

    w01 = weights[1]
    w12 = weights[2]
    
    # Train
    for i = 1:iters 
        correct_cnt = 0
        for j = 1:n_data÷batch_size
            batch_begin, batch_end = (j-1)*batch_size + 1, j * batch_size

            l0 = train_x[:, batch_begin:batch_end]
            l1 = tanh.(w01 * l0)
            dropout_mask = rand((0,1), size(l1))
            l1 = l1 .* dropout_mask .* 2
            l2 = softmax(w12 * l1)
            
            for k = 1:batch_size
                correct_cnt += (argmax(l2[:, k])==argmax(train_y[:, batch_begin+k-1]))
            end

            l2del = (train_y[:, batch_begin:batch_end] .- l2)./(batch_size * size(l2)[2])
            l1del = w12' * l2del .* tanh2deriv.(l1)

            w12 += alpha .* l2del * l1' 
            w01 += alpha .* l1del * l0'
        end

        test_correct_cnt = 0
        for j in 1:size(test_x)[2]
            l0 = test_x[:, j]
            l1 = tanh.(w01 * l0)
            l2 = w12 * l1
            test_correct_cnt += (argmax(test_y[:, j])==argmax(l2))
        end

        if(i % 10 == 0 || i == iters)
            println("Test-acc: $(test_correct_cnt/size(test_y)[2])")
            println("Train-acc: $(correct_cnt/n_data)")
        end         
    end
end

tanh2deriv(output) = 1 - output^2            
function softmax(x::Array)
    x = exp.(x)
    
    for col in 1:size(x)[2]
        x[:, col] = x[:, col] ./ sum(x[:, col])
    end
    x
end

main(hparam, metadata)
















 
                
                
                
