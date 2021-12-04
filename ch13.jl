abstract type Tensor end

#= 
    Tensor_1
        Fields: data::Array
        Functions: add
=#
mutable struct Tensor_1 <: Tensor 
    data::Array
end

+(a::Tensor_1, b::Tensor_1) = Tensor_1(A.data + B.data)

x = Tensor_1([1, 2, 3, 4, 5])

y = x + x
println(y.data)

mutable struct Tensor_2 <: Tensor
    data::Array
    creators
    creation_op
    grad
    
    Tensor_2(data; creators=nothing, creation_op=nothing) = new(data, creators, creation_op)
end

function backward(t::Tensor, grad)
    t.grad = grad
    
    if creation_op == "add"
        for creator in creators
            backward(creator, grad)
        end
    end
end

+(a::Tensor_2, b::Tensor_2) = Tensor_2(a.data+b.data, creators=[a,b], creation_op = +)

x = Tensor_v2([1,2,3,4,5])
y = Tensor_v2([2,2,2,2,2])
z = x + y
backward(z, [1,1,1,1,1])

println(x.data, y.data, z.data, z.creators, z.creators_op)

mutable struct Tensor_3 <: Tensor
    data::Array
    autograd
    creators::Vector
    creation_op::Function
    id
    children::Dict
    grad
    
    function Tensor_3(data; autograd=false, creators=nothing, creation_op=nothing, id=nothing)
        if isnothing(id)
            id = rand(1:100000)
        end
        T = new(data, autograd, creators, creation_op, id)
        T.children = Dict()
        T.grad = nothing
        
        if !(isnothing(creators))
            for creator in creators
                if haskey(creator.children, T.id)
                    creator.children[T.id] += 1
                else
                    creator.children[T.id] = 1
                end
            end
        end
        return T
    end
end

function all_children_grads_accounted_for(t::Tensor)
    for (id, cnt) in t.children
        if (cnt != 0)
            return false
        end
    end
    return true
end

function backward(t::Tensor_v3, grad=nothing, grad_origin=nothing)
    if t.autograd
        grad = Tensor_v3(ones(size(t.data)))
    
        if !(isnothing(grad_origin))
            if t.children[grad_origin.id] == 0
                throw("cannot backprop more than once")
            else
                t.children[grad_origin.id] -= 1
            end
        end
        
        if isnothing(t.grad)
            t.grad = grad
        else
            t.grad += grad
        end
        
        # grads must not have grads of their own
        @assert !grad.autograd
        
        # only continue backpropping if there's something to\n",
        # backprop into and if all gradients (from children)\n",
        # are accounted for override waiting for children if\n",
        # \"backprop\" was called on this variable directly\n",
        
        if (!isnothing(t.creators) && (all_children_grads_accounted_for(t) || isnothing(grad_origin)))
            if t.creation_op == +
                backward(t.creators[1], t.grad, t)
                backward(t.creators[2], t.grad, t)
            end
        end
    end
end






    
