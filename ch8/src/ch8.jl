using MLDatasets
using Random
using LinearAlgebra


train_x, train_y = MNIST.traindata()

test_x, test_y = MNIST.testdata()

images = reshape(train_x[:, :, 1:1000]./255, (28*28, 1000))'

labels = train_y[1:1000]

one_hot_labels = zeros(length(labels),10)

for (i, l) in enumerate(labels)
    one_hot_labels[i, l + 1] = 1
end
labels = one_hot_labels

test_img = reshape(test_x, (28*28,size(test_x)[3]))./255
test_labels = zeros(length(test_y), 10)

for (i, l) in enumerate(test_y)
    test_labels[i, l + 1] = 1
end

Random.seed!(1)

relu(x) = (x >= 0) * x
relu2deriv(x) = (x >= 0)

alpha, iters, h_size, ppi, n_labels = (0.005, 350, 40, 784, 10) # ppi = pixels per image

w01 = 0.2 * rand(ppi, h_size) .- 0.1

w12 = 0.2 * rand(h_size, n_labels) .- 0.1

for i in 1:iters
    error, correct_cnt = .0, 0

    for j in 1:size(images)[1]
        l0 = images[j,:]
        l1 = relu.(l0' * w01)
        l2 = l1 * w12

        error += sum((labels[j] .- l2) .^ 2)
        correct_cnt += (argmax(l2) == argmax(labels[j,:]))

        l2del = (labels[j] .- l2)

        l1del = (l2del * w12') .* relu2deriv.(l1) 
        global w01, w12
        
        w12 += alpha * l1' * l2del
        w01 += alpha * l0 * l1del
    end

    println("Error: $(error / float(length(images)))")
    println("Correct: $(correct_cnt/float(length(images)))")
end




















