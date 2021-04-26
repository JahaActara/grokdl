#=  Preprocessing  =#
# Get training x
# Get training y
# split the review into words using split, take a set(only 1 instance of any word),
# and make a list.
tokens = Vector(Set(split(raw_reviews, "\n")))

vocab = Set()
for sentence in tokes
    for word in sentence
        if length(word) > 0
            push!(vocab,word)
        end
    end
end
vocab = Vector(vocab)

word2index = Dict()
for index, word in enumerate(vocab)
    word2index[word] = index
end

input_dataset = Vector()
for sentence in tokens
    sent_indeces = Vector()
    for word in sentence
        try
            append(sent_indices, word2index[word])
        catch
            nothing
        end
    end
    append(input_dataset, Vector(Set(sent_indices)))
end

target_dataset = Vector()
for label in raw_labels
    if label == 'positive\n'
        push!(target_dataset, 1)
    else
        push!(target_dataset, 0)
    end
end

#=  End of preprocessing  =#

#=  Embedding Layer  =#

# Reproducibility
using Random
Random.seed!(1)

sigmoid(x) = 1/(1 + exp(-x))

# Hyperparameters
alpha, iters = 0.01, 2
h_size = 100

# Weights
w01 = 0.2 .* rand(h_size, length(vocab)) .- 0.1
w12 = 0.2 .* rand(1, h_size) .- 0.1

for iter = 1:iters
    correct, total = 0, 0
    for i = 1:length(input_dataset)-1000
        x, y = (input_dataset[i], target-dataset[i])
        l1 = sigmoid.(sum(w01[:,x]; dims=2))
        l2 = sigmoid.(w12 * l1)
        
        l2_delta = l2[1] - y # compare prediction with truth
        l1_delta = w12' * l2_delta # backprop
        w01[:,x] .-= l1_delta .* alpha
        w12 .-= l1' * l2_delta .* alpha
        
        if abs(l2_delta) < 0.5
            correct += 1
        end
        
        if i%10 == 9
            progress = string(i/length(input_dataset))
            println("Iter: $(iter) Progress: $(progress[3:4]).$(progress[5:6]% Training Accuracy: $(correct*100/total)%")
        end
    end
    println()
end

# Test
correct, total = 0, 0
for i = length(input_dataset)-1000+1:length(input_dataset)
    global correct, total
    x = input_dataset[i]
    y = target_dataset[i]
    
    l1 = sigmoid.(sum(w01[:,x]; dims=2))
    l2 = sigmoid.(w12 * l1)
    
    if abs(l2[1] - y) < 0.5
        correct += 1
    end
    total += 1
end

println("Test Accuracy: $(correct * 100 / total)%")

# Define function that finds weight similarity
function similar(target="beautiful")
    target_index = word2index[target]
    scores = Dict()
    for (word, index) in word2index
        raw_diff = w01[:, index] .- (w01[:, target_index])
        squared_diff = raw_diff .^ 2
        scores[word] = -sqrt(sum(squared_diff))
    end
    scores = sort(collect(scores), by = x -> x[2])
    return scores[end-10:end]
end

println(similar("beautiful"))
println(similar("terrible"))
