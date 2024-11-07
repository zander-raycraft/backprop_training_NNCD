import numpy as np

'''
@class: FeedForwardNode
@description: This is the node class for a feedforward neural network that finds the relationship
              between two binary numbers.
'''

class FeedForwardNode:
    """
        @about: Constructor for the node class

        @params: inputs -> int, number of inputs
        @params: outputs -> int, number of outputs
        @params: node_type -> str, type of node ('input', 'hidden', or 'output')
    """
    def __init__(self, inputs, outputs, node_type='hidden'):
        self.weights = np.random.uniform(-0.5, 0.5, inputs)
        self.outputs = np.zeros(outputs)
        self.inputs = np.zeros(inputs)  # Initialize as 1D array
        self.output = 0  # Initialize output attribute
        self.node_type = node_type

    """
        @about: Function to select and apply the activation function based on node type

        @params: node_output -> float, output of the node before applying activation
        @return: Activated output of the node
    """
    def TemplatizedActivationFunction(self, node_output):
        if self.node_type == 'input':
            return node_output
        elif self.node_type == 'hidden':
            return np.tanh(node_output)  # tanh activation for hidden layers
        elif self.node_type == 'output':
            return 1 / (1 + np.exp(-node_output))  # sigmoid for output layer
        else:
            raise ValueError("Invalid node type in network architecture")

    """
        @about: Getter for the output value
        @return: Current output value of the node
    """
    def get_output(self):
        return self.output

    """
        @about: Setter for the output value
        @params: value -> float, new output value to set
    """
    def set_output(self, value):
        self.output = value

    """
        @about: Getter for weights
        @return: Current weight vector of the node
    """
    def get_weights(self):
        return self.weights

    """
        @about: Setter for weights

        @params: new_weights -> array<double>, new weight vector to set
    """
    def set_weights(self, new_weights):
        if len(new_weights) != len(self.weights):
            raise ValueError("Incorrect size for new weight vector")
        self.weights = new_weights

    """
        @about: Calculate and set the output of the node

        @return: Activated output of the node
    """
    def calc_output(self):
        if self.node_type != 'input':
            weighted_sum = np.dot(self.inputs, self.weights)
            self.output = self.TemplatizedActivationFunction(weighted_sum)
        else:
            self.output = self.inputs[0]  # Handle the single input case for input nodes
        return self.output

    """
        @about: Set inputs for the node

        @params: inputs -> array-like, input values for the node
    """
    def set_inputs(self, inputs):
        if len(inputs) != len(self.inputs):
            raise ValueError("Incorrect size for input vector")
        self.inputs = inputs
