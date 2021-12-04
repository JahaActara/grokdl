using MLDatasets

abstract type Layer end

# Input layer
struct InputLayer <: Layer
    input::Array{Float64, 3}
    size::Tuple
end

# Convolutional layer
struct ConvLayer <: Layer
    kernels::Array{Float64,3}
    # {number of kernels, their size, stride, padding length} In total of 4 arguments
    kernel_hparam::Tuple
    size::Tuple
end

# Fully connected layer
struct FCLayer <: Layer
    weights::Array{Float64,2}
    size::Tuple
end

# Non-linear, that is, activation layer.
struct NLLayer <: Layer
    activation
    size::Tuple
end
    

mutable struct Network
    layers::Array{Layer,1}
end

function main()
    # Get data
    train_data, test_data = get_data()
    
    # Hyperparameters
    hparameters = iters, batch_size
    
    net = Network()
    
    # Initialize kernels with dimensions ((kernel_rows*kernel_cols, num_kernels)).-0.01
    
    # Initialize weights ((n_labels,h_size)) .- 0.01
    
    # train
    for j = 1:iters
        correct_cnt = 0
        for i = 1:Int(length(train_x)/batch_size)
            batch_start, batch_end = (i-1)*batch_size+1, i*batch_size
            l0 = train_x[batch_start:batch_end]
            
            sects = []
            for row_start = 1:size(l0, 2)-kernel_rows
                for col_start = 1:size(l0, 3)-kernel_cols
                    sect = l0[row_start:row_start+kernel_rows, col_start:col_start+kernel_cols]
                    append!(sects, sect)
                end
            end
        end
        expanded_input = hcat(sects)
        es = size(expanded_input)
        flattend_input = reshape(expanded_input, es[1]*es[2], -1)
        
        kernel_output = flattend_input * kernels
        
        l1 = tanh(reshape(kernel_output, es[1], -1))
        dropout_mask = rand((0,1), size(l1))
        l1 .*= dropout_mask .* 2
        l2 = softmax(w12 * l1)
        
        for k = 1:batch_size
            labelset = labels[batch_start+k]
            _inc = Int(argmax(l2[k])==argmax(labelset))
            correct_cnt += _inc
        end
        l2_delta = (labels[batch_start:batch_end]-l2)/(batch_size*size(l2)[1])
        l1_delta = l2_delta * w12' .*tanh2deriv(l1)
        l1_delta .*= dropout_mask
        w12 += alpha .* l1' * l2_delta
    end
    return nothing
end

# get_data() and relevant functions
function get_data()
    train_data = MNIST.traindata()
    test_data = MNIST.testdata()
    
    # Preprocess data
    train_data = preprocess_data(train_data, size=1000)
    test_data = preprocess_data(test_data, size="all")
    
    return (train_data, test_data)
end

function preprocess_data(data, clean="null", size="all")
    x, y = data
    
    if size == "all"
        x = x[1:end]
        y = y[1:end]
    else 
        x = x[1:size]
        y = y[1:size]
    end
    
    # One hot vectorize
    n_labels = maximum(y)
    temp = zeros(n_labels, length(y))
    for (index, label) in enumerate(y)
        temp[label,index] = 1.0
    end
    y = temp
    
    if clean == "null"
        return (x, y)
    else if clean == "flatten"
        ppi = x[1] * x[2]
        return (reshape(ppi,length(x)), y)
    end
    return nothing
end
# End of get_data() and relevant functions
            
