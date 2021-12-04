# Classification of MNIST datasets through CNN
#   Will not be too different from the implementation in the book
#   However, will make it so that the code is more readable

# Modules
using Random, MLDatasets

# Reproducibility
Random.seed!(1)

# Hyperparameters
hparam = alpha, iters, kernel_size, n_kernels, batch_size = 2, 300, (3, 3), 16, 128

# Metadata
metadata = data_size, n_data, n_labels = (28, 28), 1000, 10

# size of hidden layer
h_size = (data_size[1] - kernel_size[1] + 1) * (data_size[2] - kernel_size[2] + 1) * n_kernels

# Generic function definition
tanh2deriv(x) = 1 - x^2

function softmax(x::Matrix)
    x = exp.(x)
    for ci in 1:size(x)[2]
        x[:, ci] = x[:, ci] ./ sum(x[:, ci])
    end
    x
end
    
function main(hparam, metadata, h_size)
    # Parse parameters and metadata, etc
    # hyperparameters
    alpha, iters, kernel_size, n_kernels, batch_size = hparam

    # metadata
    data_size, n_data, n_labels = metadata

    # Load and clean data
    train_data, test_data = load_data(data_size, n_data)

    # Initialize weight
    w12 = 0.2 * rand(n_labels, h_size) .- 0.1
    
    # Initialize kernels
    k = 0.02 * rand(n_kernels, kernel_size[1] * kernel_size[2]) .- 0.01
    
    # Train Network
    k, w12 = train!(k, w12, hparam, metadata, h_size, train_data, test_data) 
end

function load_data(data_size, n_data)
    # Parse data_size
    wid, len = data_size

    ppi = wid * len

    # Load data via MLDatasets module
    train_x, train_y = MNIST.traindata()
    test_x, test_y = MNIST.testdata()

    # Reshape given data
    train_x = train_x[:, :, 1:n_data]
    train_y = one_hot(n_labels, train_y[1:n_data]) 

    test_x = test_x[:, :, 1:100]
    test_y = one_hot(n_labels, test_y[1:100])

    train_data = train_x, train_y
    test_data = test_x, test_y
    
    return train_data, test_data
end

function one_hot(n_labels, labels)
    # Initialize to zero
    one_hot_labels = zeros(n_labels, length(labels))

    # Fill
    for (index, label) in enumerate(labels)
        one_hot_labels[label+1, index] = 1
    end
    one_hot_labels
end

function train!(kernel, w12, hparam, metadata, h_size, train_data, test_data)
    # Parse hparam, etc
    alpha, iters, kernel_size, n_kernels, batch_size = hparam
    
    # Metadata
    data_size, n_data, n_labels = metadata

    # Data
    train_x, train_y = train_data
    test_x, test_y = test_data

    kernel_rows = kernel_size[1]
    kernel_cols = kernel_size[2]
    
    # train
    for i in 1:iters
        correct_cnt = 0
        for j in 1:(n_data÷batch_size)
            batch_begin, batch_end = (j-1)*batch_size+1, j*batch_size
            l0 = train_x[:, :, batch_begin:batch_end]
            
            sects = Vector()
            for row_start in 1:size(l0)[1]-kernel_rows+1
                for col_start in 1:size(l0)[2]-kernel_cols+1
                    sect = get_img_section(l0, row_start, row_start+kernel_rows-1, col_start, col_start+kernel_cols-1)
                    push!(sects, sect)
                end
            end

            cat_sects = cat(sects..., dims = 2)
            
            flattened_input = reshape(cat_sects, (prod(kernel_size), :))

            kernel_output = kernel * flattened_input

            l1 = tanh.(reshape(kernel_output, (h_size, :)))
            dropout_mask = rand((0., 1.), size(l1))
            l1 = l1 .* dropout_mask .* 2
            l2 = softmax(w12 * l1)

            for k in 1:batch_size
                correct_cnt += (argmax(train_y[:, batch_begin + k - 1]) == argmax(l2[:, k]))
            end

            l2del = (train_y[:, batch_begin:batch_end] .- l2) ./ prod(size(l2))
            l1del = w12' * l2del .* tanh2deriv.(l1)

            l1del .*= dropout_mask

            w12 += alpha * l2del * l1'
            
            k_update = reshape(l1del, size(kernel_output)) * flattened_input'
            kernel += alpha * k_update
        end
        
        println("1")

        test_correct_cnt = 0
        for j in 1:size(test_x)[3]
            l0 = test_x[:, :, j]
            sects = Vector()
            for row_start in 1:size(l0)[1]-kernel_rows+1
                for col_start in 1:size(l0)[2]-kernel_cols+1
                    sect = get_img_section(l0, row_start, row_start+kernel_rows-1, col_start, col_start+kernel_cols-1)
                    push!(sects, sect)
                end
            end
            cat_sects = cat(sects..., dims = 2)
            flattened_input = reshape(cat_sects, (prod(kernel_size), :))

            kernel_output = kernel * flattened_input
            l1 = tanh.(reshape(kernel_output, (h_size, :)))
            l2 = w12 * l1

            test_correct_cnt += (argmax(l2[:]) == argmax(test_y[:, j]))
        end

        println("2")

        if i % 1 == 0
            println("I: $i" * "Train-Acc: $(correct_cnt/n_data)" * "Test-Acc: $(test_correct_cnt/size(test_x)[3])")
        end
    end
end

function get_img_section(layer, row_from, row_to, col_from, col_to)
    section = layer[row_from:row_to, col_from:col_to, :]
end


main(hparam, metadata, h_size)
