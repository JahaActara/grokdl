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
        push!(target_dataset, true)
    else
        push!(target_dataset, false)
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
w01 = 0.2 .* rand(length()) .- 0.1


































        

      
