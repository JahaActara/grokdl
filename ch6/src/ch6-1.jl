using Random
using LinearAlgebra

Random.seed!(1)

relu(x) = (x > .0) * x

relu2deriv(x) = x > 0

alpha = 0.2

h_size = 4

streetlights = [1 0 0 1;
                0 1 0 1;
                1 1 1 1]

w_vs_s = [1 1 0 0]

w_01 = 2 * rand(Float64, (h_size, 3)) .- 1
w_12 = 2 * rand(Float64, (1, h_size)) .- 1

for iter in 1:60
    l2_err = 0
    for i in length(size(streetlights)[2])

        global w_01
        global w_12
        l0 = streetlights[:, i]
        l1 = relu.(w_01 * l0)
        l2 = dot(w_12, l1)

        l2_del = w_vs_s[i] - l2

        l2_err += sum(l2_del.^2)

        l1_del = l2_del * relu2deriv.(l1) .* w_12' 

        w_12 += (alpha .* l1' * l2_del')
        w_01 += (alpha .* l0 * l1_del')'
    end
    
    if iter % 10 == 9
        println("Error: $l2_err")
    end
end




