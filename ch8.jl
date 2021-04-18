### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ cfbed0ba-9d17-11eb-1af5-396ffe9212cd
using MLDatasets, Random

# ╔═╡ ca08d478-ef1b-4ae9-b04a-8c72f09f46f1
# Reproducibility
Random.seed!(1)

# ╔═╡ e2136d5d-fe9b-4ad9-9b93-fd71300df4c5
relu(x) = (x >= 0) * x

# ╔═╡ a957b46c-1170-4e02-922a-8cd85591c052
relu2deriv(output) = output >= 0

# ╔═╡ 98e0aff8-2c8d-46c1-a674-22602e953e58
batch_size = 100

# ╔═╡ aa6707e7-ec06-454d-9e57-3389aed96bf4
alpha, iters = 0.001, 300

# ╔═╡ b38c99b7-ca04-420c-94fb-d517caf4e5fc
ppi, n_labels, h_size = (784, 10, 100)

# ╔═╡ b15f1007-2837-4083-8e29-69d64e8d6af2
begin
	(train_x, train_y), (test_x, test_y) = MNIST.traindata(), MNIST.testdata()
	train_x = reshape(train_x[:,:,1:1000],(ppi,1000))
	train_y = 
		begin
			temp = zeros(10,length(train_y[1:1000]))
				for (i, l) in enumerate(train_y[1:1000])
					temp[l+1,i] = 1.0
				end
			temp
		end
	test_x = reshape(test_x, (ppi, size(test_x, 3)))
	test_y = 
		begin
			temp = zeros(10,length(test_y))
				for (i, l) in enumerate(test_y)
					temp[l+1,i] = 1
				end
			temp
		end
end

# ╔═╡ 63e0e75a-3db7-4514-a530-60892d7801e0
w01 = 0.2 .* rand(h_size,ppi) .- 0.1

# ╔═╡ f8853dc8-fa29-49ad-b01d-cfa42afd9b49
w12 = 0.2 .* rand(n_labels, h_size) .- 0.1

# ╔═╡ cc35ecda-fd1f-43f0-8b4c-81427d94cd10
for j in 1:iters
	error, correct_cnt = (0.0, 0)
	for i in 1:Int(length(train_x) / batch_size)
		batch_start, batch_end = ((i-1) * batch_size)+1,(i * batch_size)
		
		l0 = train_x[batch_start:batch_end]
		l1 = relu.(w01*l0')
		dropout_mask = rand((0.0,1.0), size(l1))
		l1 .*= dropout_mask .* 2
		l2 = w12 * l1
		
		error += sum((train_y[batch_start:batch_end] - l2)^2)
		for k = 1:batch_size
			correct_cnt += Int(argmax(l2[k])==argmax(train_y[batch_start + k]))
			
			l2_delta = (train_y[batch_start:batch_end]-l2)/batch_size
			
			l1_delta = l2_delta * w12' .* relu2deriv(l1)
			l1_delta *= dropout_mask
			
			w12 += alpha * l1' * l2_delta
			w01 += alpha * l0' * l1_delta
		end
		if j % 10 == 0
			test_error, test_correct_cnt = (0.0, 0)
			
			for i = 1:length(test_x)
				l0 = test_x[i]
				l1 = relu(l0 * w01)
				l2 = l1 * w12
			end
		end
	end
end

# ╔═╡ Cell order:
# ╠═cfbed0ba-9d17-11eb-1af5-396ffe9212cd
# ╠═ca08d478-ef1b-4ae9-b04a-8c72f09f46f1
# ╠═b15f1007-2837-4083-8e29-69d64e8d6af2
# ╠═e2136d5d-fe9b-4ad9-9b93-fd71300df4c5
# ╠═a957b46c-1170-4e02-922a-8cd85591c052
# ╠═98e0aff8-2c8d-46c1-a674-22602e953e58
# ╠═aa6707e7-ec06-454d-9e57-3389aed96bf4
# ╠═b38c99b7-ca04-420c-94fb-d517caf4e5fc
# ╠═63e0e75a-3db7-4514-a530-60892d7801e0
# ╠═f8853dc8-fa29-49ad-b01d-cfa42afd9b49
# ╠═cc35ecda-fd1f-43f0-8b4c-81427d94cd10
