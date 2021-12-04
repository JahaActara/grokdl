# Introduction to neural learning


function learn_single_step(w, x, y, learn_rate)
    # what the algorithm predicts from x
    yhat = w * x

    # the difference(NOT the error) between y and yhat
    delta = y - yhat
    
    # the error, defined by delta
    error = delta^2

    # Since y_hat is function of w and x, and what we need to change is w, we need to remove the influence of x from delta(from what i understand)
    w_delta = delta * x

    w = w + learn_rate * w_delta

    println("Error: $error  Prediction: $yhat")
    println("Delta: $delta   Weight Delta: * $w_delta")

    return w
end

function learn!(w, x, y, learn_rate, n)
    for i in 1:n
        w = learn_single_step(w, x, y, learn_rate)
    end

    return w
end 

## TEST ##

# Given initial parameters(weight, learn_rate, N_iters), and data(X, Y), the algorithm should converge the parameter so that W * X = Y

# Initial parameters
weight = 0.0
learn_rate = 0.1
n_iters = 10

# Data
x = 2.0
y = 0.8

println(learn!(weight, x, y, learn_rate, n_iters))

