# This is a second attempt at generalizing what I have learned and writing it for myself

# What we have: MNIST dataset which consists of training and test data. 
# Training data: 60000 images of shape 28 * 28
# Test data: unknown N of images with the same dimension
#
# GIVEN: alpha, iters, h_size, ppi, n_labels = (0.005, 350, 40, 784, 10)
#
# How it should work:
#   This problem is classification problem, given an image, the network should answer which number it belongs to.
#   It should use relu network, single image as a batch, print error while training, one hot labels,

using MLDatasets
using Random
using LinearAlgebra

Random.seed!(1)

function main()

    hparam = (0.005, 350, 40, 784, 10)

    train_x, train_y = MNIST.traindata()
    test_x, test_y = MNIST.testdata()

    # clean data
    train_x = reshape(train_x[:,:,1:1000], (28 * 28, 1000))
    train_y = one_hot(train_y[1:1000])

    test_x = reshape(test_x, (28 * 28, size(test_x)[3]))
    test_y = one_hot(test_y)
    
    # initialize and pack weight
    w01 = 0.2 .* rand(40, 784) .- 0.1
    w12 = 0.2 .* rand(10, 40) .- 0.1
    w = [w01, w12]

    w = train!(w, hparam, train_x, train_y)

    # test
    test(w, test_x, test_y)

end

function test(w, test_x, test_y)
    w01 = w[1]
    w12 = w[2]

    error, correct_cnt = (0.0, 0)

    for i in 1:size(test_x)[2]
        l0 = test_x[:,i]
        l1 = relu.(w01 * l0)
        l2 = w12 * l1

        error += sum((test_y[:,i] .- l2) .^ 2)
        correct_cnt += (argmax(l2) == argmax(test_y[:, i]))
    end
    println("Test-err: $(error/size(test_x)[2])")
    println("Test-acc: $(correct_cnt/size(test_x)[2])")
end

function one_hot(y)
    one_hot_y = zeros(10, length(y)) 
    for (i, l) in enumerate(y)
        one_hot_y[l+1, i] = 1
    end
    one_hot_y
end

function train!(w, hparam, train_x, train_y) # w is modified and returned. The function shows error during training.
    # unpack hparam
    alpha, iterations, h_size, ppi, n_labels = hparam

    for i in 1:iterations
        error_per_iter, correct_cnt_per_iter = 0.0, 0
        for j in 1:size(train_x)[2]
            img = train_x[:, j]
            lbl = train_y[:, j]

            w, error, correct_cnt = train_single_step!(w, hparam, img, lbl)
            error_per_iter += error
            correct_cnt_per_iter += correct_cnt
        end

        println("Error: $(error_per_iter / 1000)")
        println("Correct: $(correct_cnt_per_iter/1000)")
    end
    w 
end

function train_single_step!(w, hparam, img, lbl)
    # unpack hparam
    alpha, iterations, h_size, ppi, n_labels = hparam
    
    # unpack w
    w01 = w[1]
    w12 = w[2]

    l0 = img
    l1 = relu.(w01 * l0)
    l2 = w12 * l1

    error = sum((lbl .- l2) .^ 2)
    correct_cnt = argmax(l2) == argmax(lbl)

    l2del = lbl .- l2
    l1del = w12' * l2del .* relu2deriv.(l1)

    w12 += alpha * l2del * l1'
    w01 += alpha * l1del * l0'
    w = [w01, w12]
    return w, error, correct_cnt
end

relu(x) = (x > 0) * x

relu2deriv(x) = (x > 0)

main()
























