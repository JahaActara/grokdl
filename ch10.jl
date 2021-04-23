function main()
    # Get data
    
    # Preprocess data
    
    # Hyperparameters, and input rows, kernel rows, etc
    
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
        
            
