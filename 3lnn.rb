require 'nmatrix'
require 'pp'

# sigmoid chosen as non-linear function
# takes derivative if derivative == true
def nonlin(x, derivative = false)
  if derivative
    x * (-x + 1)
  else
    x.map! { |y| 1 / (1 + Math.exp(-1 * y)) }
    x
  end
end

# input data, first two terms are possible inputs, third term is bias
# is xor there are only 4 possible values if two items
INPUT_DATA = N[[0, 0, 1],
               [0, 1, 1],
               [1, 0, 1],
               [1, 1, 1]]

# output data of xor
OUTPUT_DATA = N[[0],
                [1],
                [1],
                [0]]

# seeds random
srand 1

# each sample has 2 inputs plus a bias term.
# the sample is moving to a hidden layer with 4 nodes
# therefore we need a 3x4 Matrix filleds with random weights to start
synapse_in_to_layer1 = NMatrix.new([3, 4], Array.new(6) { (2 * rand) - 1 })

# the 4 hidden layers go to one output
synapse_layer1_to_layer2 = NMatrix.new([4, 1], Array.new(6) { (2 * rand) - 1 })

# training
(1..100_000).each do |i|
  # moving forward through NN
  input_layer = INPUT_DATA
  layer1 = nonlin(input_layer.dot(synapse_in_to_layer1))
  @layer2 = nonlin(layer1.dot(synapse_layer1_to_layer2))

  # back propagation
  layer2_error = OUTPUT_DATA - @layer2

  # how often to show error rate
  puts "Error: #{layer2_error.abs.mean}" if (i % 10_000).zero?

  layer2_delta = layer2_error * nonlin(@layer2, true)
  layer1_error = layer2_delta.dot(synapse_layer1_to_layer2.transpose)
  layer1_delta = layer1_error * nonlin(layer1, true)

  # update weights
  synapse_layer1_to_layer2 += layer1.transpose.dot(layer2_delta)
  synapse_in_to_layer1 += input_layer.transpose.dot(layer1_delta)
end

puts "Output after training: #{pp @layer2}"
