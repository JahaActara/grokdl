using LinearAlgebra


weights = [0.5, 0.48, -0.7]
alpha = 0.1 # learning rate

streetlights = [[1, 0, 1],
                [0, 1, 1],
                [0, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
                [1, 0, 1]]

walk_vs_stop = [0, 1, 0, 1, 1, 0]

for iters in 1:40
    err_cumulative = 0
    for row_i in 1:length(walk_vs_stop)
        input = streetlights[row_i]
        goal_pred = walk_vs_stop[row_i]

        pred = dot(input, weights)

        delta = goal_pred - pred
        error = delta^2
        err_cumulative += error

        global weights += alpha * input * delta
        # print("Prediction: $pred")
    end
    println("Error: $err_cumulative")
end

    
