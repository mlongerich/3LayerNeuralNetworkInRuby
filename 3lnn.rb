require 'nmatrix'
require 'pp'
# This creates a 3 Layer Neural Network
class NeuralNetwork
  # input data, first three terms are votes, fourth term is bias
  # rule is majority wins
  INPUT_DATA = N[[0, 0, 0, 1],
                 [0, 1, 1, 1],
                 [0, 1, 0, 1],
                 [1, 0, 0, 1],
                 [1, 1, 0, 1],
                 [1, 1, 1, 1]]

  # output data of majority wins
  OUTPUT_DATA = N[[0],
                  [1],
                  [0],
                  [0],
                  [1],
                  [1]]

  # test data
  TEST_DATA = N[[0, 0, 1, 1],
                [1, 0, 1, 1]]

  # test data output should be
  # TEST_OUTPUT_DATA = N[[0],
  #                     [1]]

  def initialize(input = INPUT_DATA, output = OUTPUT_DATA, test = TEST_DATA)
    @input = input
    @output = output
    @test = test
    elements = @input.shape[1]
    nodes = @input.shape[1] * 2
    srand 1 # seeds random
    @synapse0 = NMatrix.new([elements, nodes], Array
                            .new(elements * nodes) { (2 * rand) - 1 })
    @synapse1 = NMatrix.new([nodes, 1], Array.new(nodes) { (2 * rand) - 1 })
  end

  def train_data(iterations = 100_000, report_percentage = 10)
    (1..iterations).each do |i|
      moving_forward(@input)
      back_propagation(iterations, i, report_percentage)
    end
    puts "Output after training: #{pp @layer2}"
  end

  def test_data(test = @test)
    moving_forward(test)
    puts "Output of tests: #{pp @layer2}"
  end

  private

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

  def moving_forward(input)
    @input_layer = input
    @layer1 = nonlin(@input_layer.dot(@synapse0))
    @layer2 = nonlin(@layer1.dot(@synapse1))
  end

  def back_propagation(iterations, i, report_percentage)
    layer2_error = @output - @layer2

    report(iterations, i, report_percentage)

    layer2_delta = layer2_error * nonlin(@layer2, true)
    layer1_error = layer2_delta.dot(@synapse1.transpose)
    layer1_delta = layer1_error * nonlin(@layer1, true)
    # update weights
    @synapse1 += @layer1.transpose.dot(layer2_delta)
    @synapse0 += @input_layer.transpose.dot(layer1_delta)
  end

  def report(iterations, i, reportp)
    puts "Error: #{layer2_error.abs.mean}" if (i % (iterations / reportp)).zero?
  end
end
