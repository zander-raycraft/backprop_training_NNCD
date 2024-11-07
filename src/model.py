from layer import Layer

'''
@class: FeedForwardModel
@description: This is the model class for a feedforward neural network that finds the relationship
              between two binary numbers using forward and backward propagation.
'''

class FeedForwardModel:
    """
        @about: Constructor for the FeedForwardModel class.

        @params: size_input -> int, number of input nodes.
        @params: size_output -> int, number of output nodes.
        @params: size_hidden -> list<int>, number of nodes in each hidden layer.
        @params: learning_rate -> float, learning rate for weight updates.
    """

    def __init__(self, size_hidden, size_input=1, size_output=2, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.input_layer = Layer(num_nodes=size_input, num_inputs=1, layer_type='input')
        self.hidden_layers = []
        prev_layer_size = size_input
        for num_nodes in size_hidden:
            hidden_layer = Layer(num_nodes=num_nodes, num_inputs=prev_layer_size, layer_type='hidden')
            self.hidden_layers.append(hidden_layer)
            prev_layer_size = num_nodes
        self.output_layer = Layer(num_nodes=size_output, num_inputs=prev_layer_size, layer_type='output')

    """
        @about: Load and format data from a text file for training and testing.

        @params: filename -> str, path to the text file containing the data.
        @return: list of tuples, formatted as (inputs, target_outputs).
    """
    def load_data(self, filename):
        data = []
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                binary1 = [int(bit) for bit in parts[0]]
                binary2 = [int(bit) for bit in parts[1]]
                result = [int(bit) for bit in parts[2]]
                inputs = binary1 + binary2
                target_outputs = result
                data.append((inputs, target_outputs))
        return data

    """
        @about: Perform a forward pass through the entire network.

        @params: inputs -> array-like, input values for the network.
        @return: list, output values from the final layer.
    """
    def forward(self, inputs):
        current_outputs = self.input_layer.forward(inputs)
        for hidden_layer in self.hidden_layers:
            current_outputs = hidden_layer.forward(current_outputs)
        return self.output_layer.forward(current_outputs)

    """
        @about: Perform a backward pass and update weights to train the network.

        @params: inputs -> array-like, input values for the network.
        @params: target_outputs -> array-like, target output values for training.
    """
    def backpropagate(self, inputs, target_outputs):
        actual_outputs = self.forward(inputs)
        output_errors = [target_outputs[i] - actual_outputs[i] for i in range(len(target_outputs))]
        delta_output = []
        for i, node in enumerate(self.output_layer.nodes):
            gradient = output_errors[i] * actual_outputs[i] * (1 - actual_outputs[i])  # Sigmoid derivative
            delta_output.append(gradient)
            for j, weight in enumerate(node.weights):
                node.weights[j] += self.learning_rate * gradient * self.hidden_layers[-1].get_outputs()[j]
        deltas = delta_output
        for i in reversed(range(len(self.hidden_layers))):
            hidden_layer = self.hidden_layers[i]
            prev_outputs = self.hidden_layers[i - 1].get_outputs() if i > 0 else inputs
            new_deltas = []
            for j, node in enumerate(hidden_layer.nodes):
                error = sum(delta * node.weights[j] for delta in deltas)
                gradient = error * (1 - node.output ** 2)  # Tanh derivative
                new_deltas.append(gradient)
                for k, weight in enumerate(node.weights):
                    node.weights[k] += self.learning_rate * gradient * prev_outputs[k]
            deltas = new_deltas

    """
        @about: Train the network on provided data using backpropagation.

        @params: training_data -> list<tuple>, each tuple is (inputs, target_outputs).
        @params: epochs -> int, number of times to iterate over the training data.
    """
    def train(self, training_data, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for inputs, target_outputs in training_data:
                self.backpropagate(inputs, target_outputs)
                actual_outputs = self.forward(inputs)
                total_loss += sum((target - actual) ** 2 for target, actual in zip(target_outputs, actual_outputs)) / 2
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(training_data)}")
