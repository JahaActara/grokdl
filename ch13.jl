abstract type Tensor end

#= 
    Tensor_1
        Fields: data::Array
        Functions: add
=#
mutable struct Tensor_1 <: Tensor 
    data::Array
end

mutable struct Tensor_2 <: Tensor
    data::Array
    
end

mutable struct Tensor_3 <: Tensor
    data::Array
    creators::Vector
    creation_op::Function
    grad::Array
    autograd::Bool
    children::Dict
    id::nothing
end

Tensor(data) = Tensor(data, [], nothing, [], false, Dict(), )

function add(t1::Tensor, t2::Tensor)::Tensor
    return Tensor(t1.data .+ t2.data, [t1, t2], add, nothing)
end

function backward(t1::Tensor, grad)::Tensor
    t1.grad = grad
    if t1.creation_op == add
        for parent in t1.creators
            backward(parent, grad)
        end
    end
end
        

    
