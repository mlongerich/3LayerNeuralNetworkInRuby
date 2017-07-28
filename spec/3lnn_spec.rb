require 'spec_helper'
require_relative '../lib/3lnn.rb'

RSpec.describe 'Neural Network' do
  let(:input) do
    N[[0, 0, 0, 1],
      [0, 1, 1, 1],
      [0, 1, 0, 1],
      [1, 0, 0, 1],
      [1, 1, 0, 1],
      [1, 1, 1, 1]]
  end
  let(:output) { N[[0], [1], [0], [0], [1], [1]] }
  let(:test) { N[[0, 0, 1, 1], [1, 0, 1, 1]] }
  let(:test_output) { N[[0], [1]] }
  let(:neural_network) { NeuralNetwork.new(input, output, test) }
  describe 'A new neural network' do
    it 'should intialize' do
      expect(neural_network).to_not be_nil
      expect(neural_network.instance_variable_get(:@input)).to_not be_nil
      expect(neural_network.instance_variable_get(:@output)).to_not be_nil
      expect(neural_network.instance_variable_get(:@test)).to_not be_nil
    end
    it 'should be able to think about the testing data before training' do
      expect(neural_network.test_data).to_not be_nil
      expect(neural_network.test_data.round).to_not eq test_output
    end
    it 'should be able to think about the input data before training' do
      expect(neural_network.test_data(input)).to_not be_nil
      expect(neural_network.test_data(input).round).to_not eq output
    end
    it 'should be able to train on the input data' do
      expect(neural_network.train_data(10_000, 1)).to_not be_nil
      expect(neural_network.train_data(10_000, 1).round).to eq output
    end
    it 'should be able to improve on what it thought about the testing data' do
      expect(neural_network.test_data.round).to_not eq test_output
      neural_network.train_data(10_000, 1)
      expect(neural_network.test_data.round).to eq test_output
    end
  end
end
