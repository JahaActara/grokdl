using Random
Random.seed!(1)

using MLDatasets
x_train, y_train = MNIST.traindata()
x_test, y_test = MNIST.testdata()

images, labels = x_train[:, :, 1:1000], y_train[1:1000]

one_hot_labels = zeros(10, length(labels))
for (i, l) in enumerate(labels)
    one_hot_labels[l+1, i] = 1.0
end
labels = one_hot_labels

test_labels = zeros(10, length(y_test))
for (i, l) in enumerate(y_test)
    test_labels[l+1, i] = 1.0
end

tanh2deriv(output) = 1 - output^2

function softmax(x)
    temp = exp.(x)
    return temp ./ sum(temp, dims=1)
end

alpha, iters = 0.2, 30
ppi, n_labels = 784, 10
batch_size = 128

input_rows, input_cols = 28, 28

kernel_rows, kernel_cols = 3, 3
num_kernels = 16

h_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels

kernels = 0.02 .* rand(num_kernels, kernel_rows*kernel_cols) .- 0.01

w12 = 0.2 .* rand(n_labels, h_size) .- 0.1

function get_img_section(layer, row_from, row_to, col_from, col_to)
    section = layer[col_from:col_to, row_from:row_to, :]
    return reshape(section, (col_to-col_from+1, row_to-row_from+1, 1, :))
end

for i = 1:iters
    correct_cnt = 0
    for j = 1:batch_size:size(images,3)-batch_size
        batch_start, batch_end = i, i+batch_size-1
        l0 = images[:,:, batch_start:batch_end]
        sects = []
        for col_start = 1:size(l0,1) - kernel_cols
            for row_start = 1:size(l0, 2) - kernel_rows
                sect = get_img_section(l0, col_start, col_start+kernel_cols-1, row_start, row_start+kernel_rows-1)
                push!(sects, sect)
            end
        end
        expanded_input = cat(sects..., dims=3)
        es = size(expanded_input)
        flattened_input = reshape(expanded_input, (:, es[3]*es[4]))
        kernel_output = kernels * flattened_input
        l1 = tanh.(reshape(kernel_output, (:, es[4])))
        dropout_mask = bitrand(size(l1))
        l1 .*= dropout_mask .* 2
        l2 = softmax(w12 * l1)

        correct_cnt += sum(argmax(l2, dims=1) .== argmax(labels[:,batch_start:batch_end], dims=1))
        l2_delta = (labels[:,batch_start:batch_end] .- l2) ./ (batch_size * size(l2, 2))
        l1_delta = w12' * l2_delta .* tanh2deriv.(l1)
        l1_delta .*= dropout_mask
        global w12 += alpha .* l2_delta * l1'
        l1d_reshape = reshape(l1_delta, size(kernel_output)) 
        k_update = l1d_reshape * flattened_input'
        kernels .-= alpha .* k_update
    end

    test_correct_cnt = 0
    
    for j=1:size(x_test, 3)
        l0 = x_test[:,:, j]
        sects = []
        for col_start=1:size(l0, 1) - kernel_cols
            for row_start=1:size(l0, 2)-kernel_rows
                sect = get_img_section(l0, col_start, col_start+kernel_cols-1, row_start, row_start+kernel_rows-1)
                push!(sects, sect)
            end
        end
        expanded_input = cat(sects...,dims=3) ##
        es = size(expanded_input)
        flattened_input = reshape(expanded_input, (:,es[3]*es[4]))
        kernel_output = kernels * flattened_input
        l1 = tanh.(reshape(kernel_output, (:, es[4])))
        dropout_mask = bitrand(size(l1))
        l1 .*= dropout_mask .* 2
        l2 = softmax(w12 * l1)
        
        test_correct_cnt += Int(argmax(dropdims(l2, dims=2)) == argmax(test_labels[:,j]))
    end
    
    if (i % 100 == 0)
        println("I: $(i) Test accuracy: $(test_correct_cnt/size(x_test, 3)) Train accuracy: $(correct_cnt/size(images, 3)) ")
    end  
end