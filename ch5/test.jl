## TEST ##

## Chapter 5

# This chapter is about using multiple inputs to predict binary state.
# The data is given as series of vectors(1d arrays), but is an array of arrays
# Type, range, length, number of the data is as follows:
# Float ranging from 0 to 1, length is 4, number is 3(3 types of data)
# Output, or y is 0 or 1, length is 4.

# Requirements
# Input is weight, x, y, learn_rate, n_iters
# Output(y) is [0.0, 1.0...] (The reason y is float is to eliminate conversion)
# The data must first be parsed and cleaned.

# Data
w = [0.1, 0.2, -0.1]

toes =  [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
x = [toes, wlrec, nfans]

y = [1.0, 1.0, 0.0, 1.0]

learn_rate = 0.1

n_iters = 5

include("ch5.jl")

# Test 1
# The input data is parsed and cleaned(into nice 2d arrays)
# Multiplication form is: y = xw, [toes_col, wlrec_col, nfans_col] * weights
# The shape of x is 4 * 3 array, length of weights should be 4
x = clean_x!(x)

@assert size(x) == (4,3) "The shape of x is correct." 

# Test 2
# The network(weights) must converge CORRECTLY, and not diverge.

w, e = train!(w, x, y, learn_rate, n_iters)

@assert e .< 0.1 "The network has converged"


