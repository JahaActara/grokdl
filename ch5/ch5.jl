function clean_x!(x)
    x = hcat(x...)
end

function train!(w, x, y, learn_rate, n_iters)
    for i in n_iters
        train_single_step!(w, x, y, learn_rate) 
    end

    e = show_error(w, x, y)
    w, e
end

function train_single_step!(w, x, y, learn_rate)
    yhat = x * w
    delta = y - yhat
    w_delta = delta .* x
    w = w + reduce(+, w_delta, dims=1)' * learn_rate
end

function show_error(w, x, y)
    (y[1] - (x[1,:]'*w))^2
end




