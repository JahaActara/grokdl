using Statistics # mean, 

# Reproducibility
using Random
Random.seed!(1)

#=  Preprocess data  =#
# Get x
f = open("revies.txt")
raw_revies = readlines(f)
close(f)

tokens = collect(map(x -> split(x, " "), raw_reviews))

vocab = Set()
for sentence in tokens
    for word in sentence
        push!(vocab, word)
    end
end
vocab = collect(vocab)
pushfirst!(vocab, "") # What does this do?

word2index = Dict()
for (i, word) in enumerate(vocab)
    word2index[word] = i
end

concat = []
input_dataset = []

for sentence in tokens
    sent_indices = []
    for word in sentence
        try
            push!(sent_indices, word2index[word])
            push!(concat, word2index[word])
        catch
            nothing
        end
    end
    push!(input_dataset, sent_indices)
end

Random.shuffle!(input_dataset)

#=  Do train  =#

alpha, iters = 0.05, 2

h_size, window, negative = 50, 2, 5

w01 = (rand(length(vocab), h_size) .- 0.5) .* 0.2 # Note that dims 1 and 2 were switched
w12 = zeros(length(vocab), h_size) # This one too

l2_target = zeros(negative+1)
l2_target[1] = 1

function similar(target="beautiful")
    target_index = word2index[target]
    scores = Dict()
    for (word, index) in word2index
        raw_diff = w01[:,index] .- w01[:,target_index]
        squared_diff = raw_diff .^ 2
        scores[word] = -sqrt(sum(squared_diff))
    end
    scores = sort(collect(scores), by = x -> x[2])
    return scores[end-10:end]
end

sigmoid(x) = 1/(1 + exp(-x))

for (rev_i, review) in enumerate(repeat(input_dataset, iters))
    for target_i = 1:length(review)
        target_samples = cat([review[target_i]], concat[floor.(Int, rand(negative) .* length(concat)) .+ 1];dims=1)
        
        left_context = review[max([1,target_i-window]):target_i-1]
        right_context = review[target_i+1:min(length(review),target_i+window)]
        
        l1 = mean(w01[left_context+right_context], axis=1)
        l2 = sigmoid.(w12[:,target_samples]' * l1)
        l2_delta = l2 - l2_target
        l1_delta = w12[:,target_samples] * l2_delta
        
        w01[:, cat(left_context, right_context;dims=1)] .-= l1_delta .* alpha
        w12[:, target_samples] .-= l2_delta' .* alpha
        println(size(w12[:,target_samples]), "    ", size(l1), size(l2_delta))
        w12[:, target_samples] .-= l1 * l2_delta' .* alpha
    end
    if (rev_i-1)%150 == 0
        progress = string(rev_i/(length(input_dataset)*iters))
        println("Iter: $(rev_i) Progress: $(progress[3:4]). $(progress[5:6])% $(similar("terrible)))
    end
end
println
print(similar("terrible"))

function analogy(positive=["terrible","good"], negative["bad"])
    norms = sum(w01 * w01, axis=1)
    normed_weights = w01 .* norms
    
    query_vect = zeros(length(w01[:,1]))
    for word in positive
        query_vect .+= normed_weights[:, word2index[word]]
    end
    for word in negative
        query_vect .-= normed_weights[:, word2index[word]]
    end
    
    scores = Dict()
    for (word, index) in word2index
        raw_diff = w01[:, index] .- query_vect
        squared_diff = raw_diff .^ 2
        scores[word] = -sqrt(sum(squared_diff))
    end
    scores = sort(collect(scores), by = x-> x[2])
    return scores[end-10:end]
end

analogy(["terrible", "good"], ["bad"])
analogy(["elizabeth", "he"], ["she"])



























