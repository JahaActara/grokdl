# Modules
using MLDatasets, Random

# Reproducibility
Random.seed!(1)

# Hyperparameters
hparam = alpha, iters, h_size, data_size, n_labels, batch_size = 0.005, 350, 100, 1000, 10, 100

function main(hparam)
    # Parse hparam
    alpha, iters, h_size, data_size, n_labels, batch_size = hparam

    # Get data
    # The load_data function cleans it up too
    train_x, train_y, test_x, test_y = load_data(ppi, data_size)
    
    # segregate training data from test data
    train_data = train_x, train_y
    test_data = test_x, test_y
    
    # Define parameters for use(ppi: pixels per image)
    # This is done by taking the first image of training data.
    ppi = reduce(*, size(train_x[1]))

    # get_weights function either loads data or initializes random weights
    weights = get_weights(ppi, h_size, n_labels)

    # trainweight! function changes weights value
    trainweights!(weights, alpha, iters, train_data, batch_size)

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

function get_weights(input_size, h_size, n_labels, prev=false)
    if prev == true
        nothing
    end

    # Initialize weights , which is a vector of arrays
    weights = Vector[]
    
    # Initial condition
    weight = init_weight(h_size, ppi)
    append!(weights, weight)

    weight = init_weight(n_labels, h_size)
    append!(weights, weight)

    weights
end

function init_weight(prev_layer_size, current_layer_size)
    0.2 * rand((current_layer_size, prev_layer_size)) - 0.1
end

function trainweight!(weights, alpha, iters, train_data, batch_size)
    # splat train_data
    train_x, train_y = train_data

    for i = 1:iters
        error, correct_cnt = 0.0, 0
        for j = 1:Int(length(train_x))/batch_size
            batch_start, batch_end = (i-1)*batch_size + 1, i * batch_size
            l0 = train_x[batch_start, batch_end]
            l1 = relu.(weights[1] * l0)
            l2 = weights[2] * l1
            error += sum(labels[batch_start:batch_end] - l2)
            for k = 1:batch_size
                correct_cnt += Int(argmax(labels[batch_start+k])==argmax(l2[k]))

                l2_delta = (labels[batch_start:batch_end] - l2)/batch_size
                l1_delta = l2_delta * weights[2]' .* relu2deriv(l1)
                weights[2] = alpha .* l1' * l2_delta
                weights[1] = alpha .* l0' * l1_delta
        end

        if(i % 10 == 0 or i == iters-1)
            test_error, test_correct_cnt = (0.0, 0)
            for j = 1:length(test_y)
                l0 = test_images[j]
                l1 = relu.(weights[1] * l0))
                l2 = weights[2] * l1
                test_error += sum((test_y[j] - l2)^2)
                test_correct_cnt += Int(argmax(l2) == argmax(test_y[j]))
                println("iters= $i, test_errors= $test_error, test_accuracy= $(test_correct_cnt/length(test_y)), train_error= $(error/length(train_x)), train_accuracy = $(correct_cnt/length(train_x))")
            end
        end
    end
end

relu(x) = (x >= 0) * x
relu2deriv(output) = (output >= 0)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
