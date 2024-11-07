from node import FeedForwardNode

'''
@class: Layer
@description: This class represents a single layer in a feedforward neural network, containing multiple nodes.
              It can be used as a modular component within the model, with forward propagation capabilities.
'''

class Layer:
    """
        @about: Constructor for the Layer class.

        @params: num_nodes -> int, number of nodes in this layer.
        @params: num_inputs -> int, number of inputs each node receives (equal to the number of nodes in the previous layer).
        @params: layer_type -> str, the type of layer ('input', 'hidden', or 'output').
    """

    def __init__(self, num_nodes, num_inputs, layer_type='hidden'):
        self.layer_type = layer_type
        self.nodes = [FeedForwardNode(inputs=num_inputs, outputs=1, node_type=layer_type) for _ in range(num_nodes)]

    """
        @about: Perform a forward pass through the layer.

        @params: inputs -> array-like, input values to be fed to each node in this layer.
        @return: list, output values from each node in this layer.
    """
    def forward(self, inputs):
        outputs = []
        if self.layer_type == 'input':
            # Assign each node one element from inputs
            for idx, node in enumerate(self.nodes):
                node.set_inputs([inputs[idx]])  # Pass a single input to each input node
                outputs.append(node.calc_output())
        else:
            # For hidden and output layers, pass the entire input vector
            for node in self.nodes:
                node.set_inputs(inputs)
                outputs.append(node.calc_output())
        return outputs

    """
        @about: Get the current outputs of all nodes in this layer.

        @return: list, current output value of each node in the layer.
    """
    def get_outputs(self):
        return [node.get_output() for node in self.nodes]

    """
        @about: Get the weights of all nodes in this layer.

        @return: list of lists, each sublist contains the weights for a single node.
    """
    def get_weights(self):
        return [node.get_weights() for node in self.nodes]

    """
        @about: Set weights for each node in the layer.

        @params: weights_matrix -> 2D array-like, each sublist is a weight vector for a node in this layer.
    """
    def set_weights(self, weights_matrix):
        if len(weights_matrix) != len(self.nodes):
            raise ValueError("Number of weight vectors must match the number of nodes.")
        for node, weights in zip(self.nodes, weights_matrix):
            node.set_weights(weights)
